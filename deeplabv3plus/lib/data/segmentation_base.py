from mxnet.gluon.data import dataset


class SegmentationDataset(dataset.Dataset):
    """Base Dataset for Segmentation
    """
    def __init__(self):
        pass

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.classes)

    @property
    def pred_offset(self):
        return 0
