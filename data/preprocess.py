import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
# Loading Libraries

import gzip
import json
import os
import shutil
from argparse import ArgumentParser

import asrpy
import mne
import numpy as np

# Loading Libraries
import pandas as pd
from braindecode.datasets.moabb import MOABBDataset
from braindecode.datasets.tuh import TUHAbnormal
from braindecode.preprocessing import (
    Pick,
    Preprocessor,  # create_fixed_length_windows, scale as multiply
    preprocess,
)
from gluonts.dataset.arrow import ArrowFile, ArrowWriter
from gluonts.dataset.common import (
    DatasetCollection,
)
from moabb.datasets.utils import dataset_list as moabb_dataset_list
from nmt import NMT
from numpy import multiply

mne.set_log_level("ERROR")  # avoid messages every time a window is extracted

_moabb_dataset_list = [dataset.__name__ for dataset in moabb_dataset_list]


def keep_unique(ds):
    # keep only unique subjects
    df = ds.description

    # Create a new column 'status' that marks duplicates as 'duplicate' and unique values as 'unique'
    df["status"] = df.duplicated("subject")
    df["status"] = df["status"].replace({True: "duplicate", False: "unique"})

    ds.set_description(df, overwrite=True)

    # Keep only one record per subject
    ds = ds.split("status")["unique"]
    return ds


def remove_common(ds):
    # Remove noisy labbled subjects
    df = ds.description
    # Split the dataframe by pathological column
    groups = df.groupby("gender")

    # Access each group as a dataframe
    df_yes = groups.get_group("F")
    df_no = groups.get_group("M")

    # Display the first five rows of each group
    df_yes.head()
    df_no.head()

    # Get the unique subjects in each group
    subjects_yes = set(df_yes["subject"].unique())
    subjects_no = set(df_no["subject"].unique())

    # Find the intersection of the two sets
    common_subjects = subjects_yes.intersection(subjects_no)

    # Display the common subjects
    print(
        f"there are {len(common_subjects)} common subjects in the two pathological groups"
    )

    # Filter the dataframe by the common subjects
    df_common = df[df["subject"].isin(common_subjects)]

    # Display the number of rows of the filtered dataframe
    print(f"There are {len(df_common)} out of {len(df)} rows have their status changed")

    # Create a new column named 'common' with boolean values
    df = df.assign(common=df["subject"].isin(common_subjects))

    # Display the first five rows of the dataframe
    # df.head()
    ds.set_description(df, overwrite=True)
    ds = ds.split("common")["False"]

    return ds


def select_by_duration(ds, tmin=0, tmax=None):
    if tmax is None:
        tmax = np.inf
    # determine length of the recordings and select based on tmin and tmax
    split_ids = []
    for d_i, d in enumerate(ds.datasets):
        duration = d.raw.n_times / d.raw.info["sfreq"]
        if tmin <= duration <= tmax:
            split_ids.append(d_i)
    splits = ds.split(split_ids)
    split = splits["0"]
    return split


def select_by_channels(ds, ch_mapping):
    split_ids = []
    for i, d in enumerate(ds.datasets):
        # print('channel names',d.raw.ch_names)
        ref = "ar" if d.raw.ch_names[0].endswith("-REF") else "le"
        # these are the channels we are looking for
        seta = set(ch_mapping[ref].keys())
        # these are the channels of the recoding
        setb = set(d.raw.ch_names)
        # if recording contains all channels we are looking for, include it
        if seta.issubset(setb):
            split_ids.append(i)
    # print(split_ids)
    return ds.split(split_ids)["0"]


def custom_crop(raw, tmin=0.0, tmax=None, include_tmax=True):
    # crop recordings to tmin – tmax. can be incomplete if recording
    # has lower duration than tmax
    # by default mne fails if tmax is bigger than duration
    tmax = min((raw.n_times - 1) / raw.info["sfreq"], tmax)
    raw.crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)


def custom_rename_channels(raw, mapping):
    # rename channels which are dependent on referencing:
    # le: EEG 01-LE, ar: EEG 01-REF
    # mne fails if the mapping contains channels as keys that are not present
    # in the raw
    if "EEG" in raw.ch_names[0]:  # just for tuh
        reference = raw.ch_names[0].split("-")[-1].lower()
        assert reference in ["le", "ref"], "unexpected referencing"
        reference = "le" if reference == "le" else "ar"
        raw.rename_channels(mapping[reference])


def custom_rename_channels_tuab(raw, mapping):
    # rename channels which are dependent on referencing:
    # le: EEG 01-LE, ar: EEG 01-REF
    # mne fails if the mapping contains channels as keys that are not present
    # in the raw
    reference = raw.ch_names[0].split("-")[-1].lower()
    assert reference in ["le", "ref"], "unexpected referencing"
    reference = "le" if reference == "le" else "ar"
    raw.rename_channels(mapping[reference])


def custom_reset_date(raw):
    # resolve this error: info["meas_date"] seconds must be between "(-2147483648, 0)" and "(2147483647, 0)"
    # print(raw.info["meas_date"])
    raw.anonymize()


def apply_asr(raw):
    try:
        # filter the data between 1 and 75 Hz
        raw.load_data()
        raw.filter(
            l_freq=1.0, h_freq=None, fir_design="firwin", skip_by_annotation="edge"
        )
        # run asr
        asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=5)
        asr.fit(raw.copy())
        raw = asr.transform(raw.copy())
    except Exception:
        print("Could not apply the ASR")
        pass


def normalize_one_recording_channel_wise(clean_eeg_data):
    # raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=True)
    short_ch_names = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    ch_names = short_ch_names  # raw.ch_names #

    # compute stats only on clean segments
    means = []
    stds = []
    for i in range(len(ch_names)):
        means.append(np.mean(clean_eeg_data[i, :]))
        stds.append(np.std(clean_eeg_data[i, :]))

    # apply z-score normalization to clean_data
    # normalized_clean_eeg_data = []
    for i in range(len(ch_names)):
        #   clip to 2 stds
        clean_eeg_data[i, :] = np.clip(
            clean_eeg_data[i, :],
            a_min=means[i] - 2 * stds[i],
            a_max=means[i] + 2 * stds[i],
        )
        #   z-score normalization
        # clean_eeg_data[i, :] = (clean_eeg_data[i, :] - means[i]) / stds[i] # zscoring
        clean_eeg_data[i, :] = (
            clean_eeg_data[i, :] - np.mean(clean_eeg_data[i, :])
        ) / np.std(clean_eeg_data[i, :])  # zscoring

    return clean_eeg_data


def create_pp_folder(dir_name):
    # Check if directory already exists
    if os.path.exists(dir_name):
        # Remove directory if it already exists
        shutil.rmtree(dir_name)

    # Create new directory
    os.mkdir(dir_name)


def dump_jsonl_gzip_file(data, file_path):
    # Convert data to JSON lines format
    jsonl_data = [json.dumps(item) for item in data]
    # Join JSON lines with newlines
    jsonl_content = "\n".join(jsonl_data)
    # Compress the JSON lines data and write it to the gzip file
    with gzip.open(file_path, "wt") as f:
        f.write(jsonl_content)


def load_jsonl_gzip_file(file_path):
    with gzip.open(file_path, "rt") as f:
        return [json.loads(line) for line in f]


def general_preprocessing(selected_ds, PATH_pp, N_JOBS):
    tmin = 5 * 60
    tmax = None
    selected_ds = select_by_duration(selected_ds, tmin, tmax)

    remove_common_flag = False
    if remove_common_flag:
        selected_ds = keep_unique(selected_ds)
        # print(len(selected_ds.description))
        selected_ds = remove_common(selected_ds)
        # print(len(selected_ds.description))

    short_ch_names = sorted(
        [
            "A1",
            "A2",
            "FP1",
            "FP2",
            "F3",
            "F4",
            "C3",
            "C4",
            "P3",
            "P4",
            "O1",
            "O2",
            "F7",
            "F8",
            "T3",
            "T4",
            "T5",
            "T6",
            "FZ",
            "CZ",
            "PZ",
        ]
    )
    if dataset == "nmt":
        ar_ch_names = sorted(
            [
                "EEG A1-REF",
                "EEG A2-REF",
                "EEG FP1-REF",
                "EEG FP2-REF",
                "EEG F3-REF",
                "EEG F4-REF",
                "EEG C3-REF",
                "EEG C4-REF",
                "EEG P3-REF",
                "EEG P4-REF",
                "EEG O1-REF",
                "EEG O2-REF",
                "EEG F7-REF",
                "EEG F8-REF",
                "EEG T3-REF",
                "EEG T4-REF",
                "EEG T5-REF",
                "EEG T6-REF",
                "EEG FZ-REF",
                "EEG CZ-REF",
                "EEG PZ-REF",
            ]
        )
        le_ch_names = sorted(
            [
                "EEG A1-LE",
                "EEG A2-LE",
                "EEG FP1-LE",
                "EEG FP2-LE",
                "EEG F3-LE",
                "EEG F4-LE",
                "EEG C3-LE",
                "EEG C4-LE",
                "EEG P3-LE",
                "EEG P4-LE",
                "EEG O1-LE",
                "EEG O2-LE",
                "EEG F7-LE",
                "EEG F8-LE",
                "EEG T3-LE",
                "EEG T4-LE",
                "EEG T5-LE",
                "EEG T6-LE",
                "EEG FZ-LE",
                "EEG CZ-LE",
                "EEG PZ-LE",
            ]
        )
    else:
        ar_ch_names = sorted(
            [
                "A1-REF",
                "A2-REF",
                "FP1-REF",
                "FP2-REF",
                "F3-REF",
                "F4-REF",
                "C3-REF",
                "C4-REF",
                "P3-REF",
                "P4-REF",
                "O1-REF",
                "O2-REF",
                "F7-REF",
                "F8-REF",
                "T3-REF",
                "T4-REF",
                "T5-REF",
                "T6-REF",
                "FZ-REF",
                "CZ-REF",
                "PZ-REF",
            ]
        )
        le_ch_names = sorted(
            [
                "A1-LE",
                "A2-LE",
                "FP1-LE",
                "FP2-LE",
                "F3-LE",
                "F4-LE",
                "C3-LE",
                "C4-LE",
                "P3-LE",
                "P4-LE",
                "O1-LE",
                "O2-LE",
                "F7-LE",
                "F8-LE",
                "T3-LE",
                "T4-LE",
                "T5-LE",
                "T6-LE",
                "FZ-LE",
                "CZ-LE",
                "PZ-LE",
            ]
        )
    assert len(short_ch_names) == len(ar_ch_names) == len(le_ch_names)
    ar_ch_mapping = {
        ch_name: short_ch_name
        for ch_name, short_ch_name in zip(ar_ch_names, short_ch_names)
    }
    le_ch_mapping = {
        ch_name: short_ch_name
        for ch_name, short_ch_name in zip(le_ch_names, short_ch_names)
    }
    ch_mapping = {"ar": ar_ch_mapping, "le": le_ch_mapping}
    print(ch_mapping)
    if "tu" in dataset:
        selected_ds = select_by_channels(selected_ds, ch_mapping)
    print("description", len(selected_ds.description))
    print("description", selected_ds.description)
    print("ch_names", selected_ds.datasets[-1].raw.info["ch_names"])
    tmin = 1 * 60
    tmax = 6 * 60
    sfreq = 100
    if dataset == "nmt":
        preprocessors = [
            Preprocessor(
                custom_crop,
                tmin=tmin,
                tmax=tmax,
                include_tmax=False,
                apply_on_array=False,
            ),
            Preprocessor(
                custom_rename_channels, mapping=ch_mapping, apply_on_array=False
            ),
            Preprocessor("pick_channels", ch_names=short_ch_names, ordered=True),
            Preprocessor(
                lambda data: multiply(data, 1 * 1e5), apply_on_array=True
            ),  # Convert from V to uV
            Preprocessor(custom_reset_date, apply_on_array=False),
            # Preprocessor(np.clip, a_min=-800, a_max=800, apply_on_array=True),
            Preprocessor("resample", sfreq=sfreq),
            Preprocessor("filter", l_freq=1, h_freq=45),
        ]
    elif dataset == "moabb":
        preprocessors = [
            Pick(picks=["eeg"]),
            Preprocessor(
                custom_crop,
                tmin=tmin,
                tmax=tmax,
                include_tmax=False,
                apply_on_array=False,
            ),
            Preprocessor(
                lambda data: multiply(data, 1 * 1e5), apply_on_array=True
            ),  # Convert from V to uV
            Preprocessor("resample", sfreq=sfreq),
            Preprocessor("filter", l_freq=1, h_freq=45),
        ]
    else:  # for tuab dataset
        preprocessors = [
            Preprocessor(
                custom_crop,
                tmin=tmin,
                tmax=tmax,
                include_tmax=False,
                apply_on_array=False,
            ),
            Preprocessor(
                custom_rename_channels_tuab, mapping=ch_mapping, apply_on_array=False
            ),
            Preprocessor("pick_channels", ch_names=short_ch_names, ordered=True),
            Preprocessor(
                lambda data: multiply(data, 1 * 1e6), apply_on_array=True
            ),  # Convert from V to uV
            Preprocessor(custom_reset_date, apply_on_array=False),
            Preprocessor(np.clip, a_min=-800, a_max=800, apply_on_array=True),
            Preprocessor("resample", sfreq=sfreq),
        ]
    create_pp_folder(PATH_pp)
    selected_preproc = preprocess(
        concat_ds=selected_ds,
        preprocessors=preprocessors,
        n_jobs=N_JOBS,
        save_dir=PATH_pp,
        overwrite=False,
    )
    print("ch_names", selected_ds.datasets[-1].raw.info["ch_names"])
    return selected_preproc


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--start_range", type=int, default=0, help="Start range of the dataset"
    )
    parser.add_argument(
        "--end_range", type=int, default=500, help="End range of the dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nmt",
        choices=["tuab", "nmt", "moabb"],
        help="Dataset to be used",
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default=".data/experiments/",
    )
    parser.add_argument(
        "--tuab_raw_path",
        type=str,
        default=".data/TUAB/v3.0.1/edf",
    )
    parser.add_argument(
        "--nmt_raw_path", type=str, default="./data/NMT/nmt_raw/nmt_scalp_eeg_dataset/"
    )
    parser.add_argument(
        "--moabb_dataset", type=str, default=None, choices=_moabb_dataset_list
    )
    args = parser.parse_args()

    Exp_Path = args.exp_path
    TUAB_raw_path = args.tuab_raw_path
    NMT_raw_path = args.nmt_raw_path
    TUAB_pp_path = Exp_Path + "/tuab/tuab_pp"
    NMT_pp_path = Exp_Path + "/NMT/nmt_pp"
    RESULTS_path = Exp_Path + "/results/"
    N_JOBS = 1  # specify the number of jobs for loading and windowing

    os.makedirs(RESULTS_path, exist_ok=True)

    dataset = args.dataset  # specify the dataset to be used 'tuab' OR 'nmt'. You need to do this before running the next cell for both datasets

    if dataset == "tuab":
        os.makedirs(TUAB_raw_path, exist_ok=True)
        os.makedirs(TUAB_pp_path, exist_ok=True)
        tuh_ds = TUHAbnormal(
            TUAB_raw_path,
            target_name=("pathological", "age", "gender"),
            recording_ids=range(args.start_range, args.end_range),
            # recording_ids=range(100),#or None to load the whole dataset,
            preload=False,
            # n_jobs=1 if TUHAbnormal.__name__ == '_TUHAbnormalMock' else N_JOBS
        )
        print(tuh_ds.description)
        selected_ds = tuh_ds
        PATH_pp = TUAB_pp_path
    elif dataset == "nmt":
        os.makedirs(NMT_raw_path, exist_ok=True)
        os.makedirs(NMT_pp_path, exist_ok=True)
        nmt_ds = NMT(
            NMT_raw_path,
            target_name=("pathological", "age", "gender"),
            recording_ids=range(args.start_range, args.end_range),
            # recording_ids=range(100,200),#or None to load the whole dataset,
            preload=False,
            # n_jobs=N_JOBS
        )
        nmt_ds.description
        selected_ds = nmt_ds
        PATH_pp = NMT_pp_path
    elif dataset == "moabb":
        PATH_pp = args.exp_path + "/moabb/" + args.moabb_dataset + "_pp"
        os.makedirs(PATH_pp, exist_ok=True)
        selected_ds = MOABBDataset(
            dataset_name=args.moabb_dataset,
            subject_ids=None,  # [i for i in range(args.start_range, args.end_range)],
        )

    selected_preproc = general_preprocessing(selected_ds, PATH_pp, N_JOBS)

    if dataset == "moabb":
        # add train and test split to the dataset
        target = selected_preproc.description["subject"]  # .astype(int)
        for i, (d, y) in enumerate(zip(selected_preproc.datasets, target)):
            if i < 0.8 * len(selected_preproc):
                d.description["train"] = True
            else:
                d.description["train"] = False
        selected_preproc.set_description(
            pd.DataFrame([d.description for d in selected_preproc.datasets]),
            overwrite=True,
        )
    train_ds = selected_preproc.split("train")["True"]
    print("selected_preproc", selected_preproc.split("train"))

    # print('ch_names',selected_ds.datasets[-1].raw.info['ch_names'])
    raw = selected_preproc.datasets[0].raw

    if dataset == "moabb":
        foldername = args.exp_path + "/moabb/" + args.moabb_dataset + "_dl"
    else:
        foldername = args.exp_path + "/" + args.dataset + "_dl"
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        os.makedirs(foldername + "/train")
        os.makedirs(foldername + "/val")
        os.makedirs(foldername + "/test_indist")
        os.makedirs(foldername + "/test")
    with open(foldername + "/metadata.json", "w") as f:
        json.dump({"freq": "10L"}, f)
    dataset_gluon_train = []
    dataset_gluon_test = []
    dataset_gluon_val = []
    dataset_gluon_testindist = []

    for i, data in enumerate(train_ds.datasets):
        for j in range(data.raw.get_data().shape[0]):
            dataset_gluon_train.append(
                {
                    "item_id": str(i),
                    "channel_id": str(j),
                    "start": str(pd.Period("2016-07-01 00:00", freq="10L")),
                    "target": data.raw.get_data()[j][:-6000].tolist(),
                }
            )

            train_series_len = len(data.raw.get_data()[j][:-6000].tolist())
            dataset_gluon_val.append(
                {
                    "item_id": str(i),
                    "channel_id": str(j),
                    "start": str(
                        pd.Period("2016-07-01 00:00", freq="10L") + train_series_len
                    ),
                    "target": data.raw.get_data()[j][-6000:-3000].tolist(),
                }
            )
            dataset_gluon_testindist.append(
                {
                    "item_id": str(i),
                    "channel_id": str(j),
                    "start": str(
                        pd.Period("2016-07-01 00:00", freq="10L") + train_series_len
                    ),
                    "target": data.raw.get_data()[j][-6000:].tolist(),
                }
            )

    filename = foldername + "/train/train_data_" + str(args.end_range) + ".arrow"
    if not os.path.exists(filename):
        ArrowWriter(suffix=".arrow").write_to_file(dataset_gluon_train, filename)

    filename = foldername + "/val/val_data_" + str(args.end_range) + ".arrow"
    if not os.path.exists(filename):
        ArrowWriter(suffix=".arrow").write_to_file(dataset_gluon_val, filename)

    filename = (
        foldername + "/test_indist/test_indist_data_" + str(args.end_range) + ".arrow"
    )
    if not os.path.exists(filename):
        ArrowWriter(suffix=".arrow").write_to_file(dataset_gluon_testindist, filename)

    if "False" in selected_preproc.split("train"):
        test_ds = selected_preproc.split("train")["False"]

        for i, data in enumerate(test_ds.datasets):
            for j in range(data.raw.get_data().shape[0]):
                dataset_gluon_test.append(
                    {
                        "item_id": str(i),
                        "channel_id": str(j),
                        "start": str(pd.Period("2016-07-01 00:00", freq="10L")),
                        "target": data.raw.get_data()[j].tolist(),
                    }
                )

        filename_test = foldername + "/test/test_data_" + str(args.end_range) + ".arrow"
        if not os.path.exists(filename_test):
            ArrowWriter(suffix=".arrow").write_to_file(
                dataset_gluon_test, filename_test
            )

    train_ds = DatasetCollection(
        datasets=[
            ArrowFile(path=foldername + "/train/" + filename)
            for filename in os.listdir(foldername + "/train/")
        ]
    )

    print("train_ds", len(train_ds))
    for x in train_ds:
        print(x)
        break
    print(f"Finished preprocessing {args.dataset} dataset. Files saved in {PATH_pp}")
    print(f"Completed {args.dataset} dataset. Arrow files saved in {foldername}")
