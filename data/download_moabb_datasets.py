from argparse import ArgumentParser
from warnings import warn

from braindecode.datasets import MOABBDataset
from moabb.datasets.utils import dataset_list
from moabb.utils import set_download_dir


def download_moabb_datasets(dataset_name, dataset_kwargs):
    """
    Download the specified MOABB dataset.
    """

    MOABBDataset(
        dataset_name=dataset_name, subject_ids=None, dataset_kwargs=dataset_kwargs
    )


if __name__ == "__main__":
    # get args
    """
    ['AlexMI',
    'BI2012',
    'BI2013a',
    'BI2014a',
    'BI2014b',
    'BI2015a',
    'BI2015b',
    'BNCI2014_001',
    'BNCI2014_002',
    'BNCI2014_004',
    'BNCI2014_008',
    'BNCI2014_009',
    'BNCI2015_001',
    'BNCI2015_003',
    'BNCI2015_004',
    'CastillosBurstVEP100',
    'CastillosBurstVEP40',
    'CastillosCVEP100',
    'CastillosCVEP40',
    'Cattan2019_PHMD',
    'Cattan2019_VR',
    'Cho2017',
    'EPFLP300',
    'GrosseWentrup2009',
    'HeadMountedDisplay',
    'Hinss2021',
    'Huebner2017',
    'Huebner2018',
    'Kalunga2016',
    'Lee2019_ERP',
    'Lee2019_MI',
    'Lee2019_SSVEP',
    'MAMEM1',
    'MAMEM2',
    'MAMEM3',
    'MunichMI',
    'Nakanishi2015',
    'Ofner2017',
    'PhysionetMI',
    'Rodrigues2017',
    'SSVEPExo',
    'Schirrmeister2017',
    'Shin2017A',
    'Shin2017B',
    'Sosulski2019',
    'Stieger2021',
    'Thielen2015',
    'Thielen2021',
    'VirtualReality',
    'Wang2016',
    'Weibo2014',
    'Zhou2016',]
    """
    dataset_list_name = [dataset.__name__ for dataset in dataset_list]
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="~/mne_data/")
    parser.add_argument("--datasets", type=str)
    args = parser.parse_args()

    if args.datasets in ["Shin2017A", "Shin2017B"]:
        dataset_kwargs = {"accept": True}
    else:
        dataset_kwargs = None
    try:
        set_download_dir(args.path)
    except Exception as ex:
        warn(f"Path already set. Continuing... {ex}")
    download_moabb_datasets(args.datasets, dataset_kwargs=dataset_kwargs)

    print("done")
