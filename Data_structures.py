import os
import glob
import numpy as np
import h5py as h5
import ForceMetric as fm
from Tools import concat_3dimages, concat_3dimages_corners
from PIL import Image


def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/',
                                                    item)
        else:
            raise ValueError('Cannot save %s type' % type(item))


def recursively_load_dict_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5._hl.group.Group):
            ans[key] = recursively_load_dict_from_group(h5file,
                                                        path + key + '/')
    return ans


def PositionFromFolder(path, file_types=('ibw', 'tif'), no_files=None,
                       skip=None):
    data = np.sort(glob.glob(path))

    if skip:
        mask = np.ones(len(data), dtype=bool)
        for s in skip:
            mask[s] = 0

        data = np.array(data)[mask]

    template = np.array([d.split('.')[-1] for d in data])
    masks = [template == ends for ends in file_types]
    mask = np.array([any(m) for m in zip(*masks)])
    data = data[mask]

    if no_files:
        data = data[:no_files]

    return data


def IBW2HDF5(inpath, data_name='', outpath=None):
    fnb = os.path.basename(inpath).split('.')[0]
    wave = fm.Wave(inpath)
    outpath = os.path.join(os.path.dirname(inpath), fnb + '.hdf5')

    h5file = h5.File(outpath, 'a')
    data_grp = h5file.create_group('data/%s' % data_name)

    for key in wave.keys():
        data_grp.attrs[key] = wave[key]

    for label in wave.labels:
        print(label)
        dt = 1 * wave.getData(label)
        data_grp.create_dataset(label, data=dt, dtype='float')
    return h5file


def hdf5_image(grp, img, no):
    # dset = grp.create_dataset(name, data=img)
    dset = grp.create_dataset('image%s' % str(no).zfill(2), data=img,
                              compression="gzip")
    dset.attrs["CLASS"] = np.string_("IMAGE")

    if len(img.shape) == 3 and (img.shape[0] == 3 or img.shape[2] == 3):
        dset.attrs["IMAGE_SUBCLASS"] = np.string_("IMAGE_TRUECOLOR")
        dset.attrs["IMAGE_COLORMOEL"] = np.string_("RGB")

        if img.shape[0] == 3:
            # Stored as [pixel components][height][width]
            dset.attrs["INTERLACE_MODE"] = np.string_("INTERLACE_PLANE")

        else:  # This is the np standard
            # Stored as [height][width][pixel components]
            dset.attrs["INTERLACE_MODE"] = np.string_("INTERLACE_PIXEL")

    else:
        dset.attrs["IMAGE_SUBCLASS"] = np.string_("IMAGE_GRAYSCALE")
        dset.attrs["IMAGE_WHITE_IS_ZERO"] = np.array(0, dtype="uint8")
        dset.attrs["IMAGE_MINMAXRANGE"] = [img.min(), img.max()]

    dset.attrs["DISPLAY_ORIGIN"] = np.string_("UL")  # not rotated
    dset.attrs["IMAGE_VERSION"] = np.string_("1.2")


def hdf5_curve(grp, curve, no):
    dat = np.rollaxis(curve.data, 0, 2)
    dset = grp.create_dataset('curve%s' % str(no).zfill(2), data=dat,
                              compression="gzip")
    grp.create_dataset('label%s' % str(no).zfill(2),
                       data=np.array(curve.labels, dtype='S20'))
    for key in curve.keys():
        dset.attrs[key] = curve[key]


def hdf5_scan(grp, scan, no):
    dat = np.rollaxis(scan.data, 0, 3)
    dset = grp.create_dataset('scan%s' % str(no).zfill(2), data=dat,
                              compression="gzip")
    grp.create_dataset('label%s' % str(no).zfill(2),
                       data=np.array(scan.labels, dtype='S20'))
    for key in scan.keys():
        dset.attrs[key] = scan[key]


def hdf5_tile(grp, tile, no):
    dat = np.rollaxis(tile.tiles, 0, 3)
    dset = grp.create_dataset('tile%s' % str(no).zfill(2), data=dat,
                              compression="gzip")
    grp.create_dataset('label%s' % str(no).zfill(2),
                       data=np.array(tile.header['labels'], dtype='S20'))
    # for key in scan.keys():
        # dset.attrs[key] = scan[key]


def hdf5_position(grp, pos, no):
    pos_grp = grp.create_group('position%s' % str(no).zfill(2))
    if len(pos.scans):
        scan_grp = pos_grp.create_group('Scans')

        for i, scan in enumerate(pos.scans):
            hdf5_scan(scan_grp, scan, i)

    if len(pos.force_curves):
        curve_grp = pos_grp.create_group('Curves')

        for i, curve in enumerate(pos.force_curves):
            hdf5_curve(curve_grp, curve, i)

    if len(pos.images):
        img_grp = pos_grp.create_group('Images')

        for i, img in enumerate(pos.images):
            hdf5_image(img_grp, img, i)


def hdf5_analysis(grp, data, ana_type):
    recursively_save_dict_contents_to_group(grp, ana_type + '/', data)


class Project:
    def __init__(self, file_name):
        self.type = 'project'
        self.path = file_name
        self.experiments = []

    def AddExperiment(self, experiment):
        self.experiments.append(experiment)


class Experiment:
    def __init__(self, file_name):
        self.type = 'experiment'
        self.path = file_name
        self.samples = []
        self.analysis = {}

    def AddSample(self, sample):
        self.samples.append(sample)

    def AddAnalysis(self, data, group):
        self.analysis[group] = data


class Sample:
    def __init__(self, file_name, name='Sample01', load=False):
        self.index = -1
        self.counter = -1
        self.path = file_name
        self.name = name
        self.type = 'sample'
        self.positions = []
        self.analysis = {}

        if load:
            f = h5.File(self.path, 'a')

            positions = f.require_group('Positions')

            for key in positions.keys():
                print('load %s' % key)
                print(positions)
                tmp = positions[key]
                # print(tmp.file)
                pos = Position(key, load='from_group', grp=tmp)
                self.positions.append(pos)

            tiles = f.require_group('Tiles')

            for key in tiles.keys():
                if 'tile' in key:
                    pth = tiles[key].name
                    lbl_path = pth.replace('tile', 'label')
                    tile = np.array(f[pth])
                    self.tiles = tile
                    self.header = list(np.array(list(f[lbl_path]),
                                                dtype='U20'))

            ana = f.require_group('Analysis')

            if ana:
                self.analysis = recursively_load_dict_from_group(ana,
                                                                  '/Analysis/')

            f.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == self.counter:
            self.index = -1
            raise StopIteration
        self.index += 1
        return self.positions[self.index]

    def AddPosition(self, position):
        self.counter += 1
        self.positions.append(position)
        if self.positions[-1].name is None:
            self.positions[-1].name = 'Position%s' % str(self.counter).zfill(4)

    def AddAnalysis(self, data, group):
        self.analysis[group] = data

    def AddData(self, data, auto=True, scan_type='DynamicMechanic'):
        if auto:
            ext = data.split('.')[-1]
            if ext in ['tif', 'jpg', 'png']:
                dtype = 'Image'
            elif ext == 'ibw':
                dtype = fm.IdentifyScanMode(data)

        if dtype == 'Image':
            img = Image.open(data)
            self.AddImage(np.array(img))
        elif dtype == 'Imaging':
            if scan_type == 'DynamicMechanic':
                scan = fm.DynamicMechanicAFMScan()
                scan.load(data)
                scan.CalcAllViscoelastic()
            else:
                scan = fm.AFMScan(data)
            self.AddScan(scan)
        elif dtype == 'ForceCurve':
            fc = fm.ForceCurve(data)
            self.AddForceCurve(fc)

    def Write(self):
        """Writes to self.path as HDF5 structure"""
        if os.path.exists(self.path):
            print("The file %s already exists." % self.path)
            check = input("Do you want to overwrite this file? [y, n] ")
            if check == 'y':
                os.remove(self.path)
            else:
                print("File is not saved")
                return
        f = h5.File(self.path, 'a')

        positions = f.create_group('Positions')

        for i, pos in enumerate(self.positions):
            hdf5_position(positions, pos, i)

        if self.analysis:
            ana = f.create_group('Analysis')
            for ana_type in self.analysis.keys():
                hdf5_analysis(ana, self.analysis[ana_type], ana_type)

        if self.type == 'tiles':
            tiles = f.create_group('Tiles')
            hdf5_tile(tiles, self, 0)

        f.flush()
        f.close()


class Tiles(Sample):
    def __init__(self, file_name, name='Sample01', load=False):
        Sample.__init__(self, file_name, name, load)
        self.type = 'tiles'

    def AddOffset(self, coord):
        for i, pos in enumerate(self.positions):
            pos.header['tile_offset'] = coord[i]

    def CreateTiles(self, transpose=True, center_offset=True, ontop=True):
        for i, pos in enumerate(self.positions):
            dat = pos.scans[0].data

            coord = pos.header['tile_offset']

            if i == 0:
                Dat = dat

            else:
                coords = list(coord)
                coords.extend([0])
                Dat = concat_3dimages_corners(Dat, dat, *coords,
                                              transpose=transpose,
                                              center_offset=center_offset,
                                              ontop=ontop)

        self.tiles = Dat
        self.header = {}
        self.header['labels'] = self.positions[0].scans[0].labels


class Position:
    def __init__(self, file_name, name=None, load=False, grp=None):
        self.type = 'position'
        self.path = file_name
        self.name = name
        self.scans = []
        self.force_curves = []
        self.images = []
        self.analysis = {}
        self.header = {}

        if load == 'from_file':
            f = h5.File(self.path, 'a')

            curves = f.require_group('Curves')

            for key in curves.keys():
                if 'curve' in key:
                    pth = curves[key].name
                    fc = fm.ForeCurve(pth, basefile=f)
                    self.force_curves.append(fc)

            scans = f.require_group('Scans')

            for key in scans.keys():
                if 'scan' in key:
                    pth = scans[key].name
                    scan = fm.AFMScan(pth, basefile=f)
                    self.scans.append(scan)

            imgs = f.require_group('Images')

            for key in imgs.keys():
                if 'image' in key:
                    pth = imgs[key].name
                    img = np.array(f[pth])
                    self.images.append(img)

            ana = f.require_group('Analysis')

            for key in ana.keys():
                self.analysis[key] = ana[key]

            f.close()

        elif load == 'from_group':
            f = grp.file

            curves = grp.require_group('Curves')

            for key in curves.keys():
                if 'curve' in key:
                    pth = curves[key].name
                    fc = fm.ForceCurve(pth, basefile=f)
                    self.force_curves.append(fc)

            scans = grp.require_group('Scans')

            for key in scans.keys():
                if 'scan' in key:
                    pth = scans[key].name
                    scan = fm.AFMScan(pth, basefile=f)
                    self.scans.append(scan)

            imgs = grp.require_group('Images')

            for key in imgs.keys():
                if 'image' in key:
                    pth = imgs[key].name
                    img = np.array(f[pth])
                    self.images.append(img)

            ana = grp.require_group('Analysis')

            for key in ana.keys():
                self.analysis[key] = ana[key]

    def AddScan(self, scan):
        self.scans.append(scan)

    def AddForceCurve(self, force_curve):
        self.force_curves.append(force_curve)

    def AddImage(self, img):
        self.images.append(img)

    def AddData(self, data, auto=True, scan_type='DynamicMechanic'):
        if auto:
            ext = data.split('.')[-1]
            if ext in ['tif', 'jpg', 'png']:
                dtype = 'Image'
            elif ext == 'ibw':
                dtype = fm.IdentifyScanMode(data)

        if dtype == 'Image':
            img = Image.open(data)
            self.AddImage(np.array(img))
        elif dtype == 'Imaging':
            if scan_type == 'DynamicMechanic':
                scan = fm.DynamicMechanicAFMScan()
                scan.load(data)
                scan.CalcAllViscoelastic()
            else:
                scan = fm.AFMScan(data)
            self.AddScan(scan)
        elif dtype == 'ForceCurve':
            fc = fm.ForceCurve(data)
            self.AddForceCurve(fc)

    def AddAnalysis(self, data, group):
        self.analysis[group] = data

    def Write(self):
        """Writes to self.path as HDF5 structure"""
        if os.path.exists(self.path):
            print("The file %s already exists." % self.path)
            check = input("Do you want to overwrite this file? [y, n] ")
            if check == 'y':
                os.remove(self.path)
            else:
                print("File is not saved")
                return
        f = h5.File(self.path, 'a')
        if len(self.scans):
            scans = f.create_group('Scans')
            for i, scan in enumerate(self.scans):
                hdf5_scan(scans, scan, i)

        if len(self.force_curves):
            curves = f.create_group('ForceCurves')
            for i, fc in enumerate(self.force_curves):
                hdf5_curve(curves, fc, i)

        if len(self.images):
            imgs = f.create_group('Images')
            for i, img in enumerate(self.images):
                name = 'image%s' % str(i).zfill(2)
                hdf5_image(imgs, img, name)

        f.flush()
        f.close()


class ComplexDataSet:
    def __init__(self, file_name):
        self.data_set = h5.File(file_name)
        self.path = file_name

    def view(self):
        self.data_set.visit()

    def close(self):
        self.data_set.close()

    def add_file(self, file_name, group='scan'):
        ext = file_name.split('.')[-1]
        if ext == 'hdf5':
            tmp = h5.File(file_name)
            if group in self.data_set:
                print("%s exists" % group)
                grp = self.data_set[group]
            else:
                grp = self.data_set.create_group(group)

            for k in tmp.keys():
                tmp.copy(k, grp)

    def add_data(self, name, data, group='Data'):
        if group in self.data_set:
            print("%s exists" % group)
            grp = self.data_set[group]
        else:
            grp = self.data_set.create_group(group)

        grp.create_dataset(name=name, data=data)

    def add_image(self, name, img, group='Data/images'):
        if group in self.data_set:
            print("%s exists" % group)
            grp = self.data_set[group]
        else:
            grp = self.data_set.create_group(group)
        ds = grp.create_dataset(name=name, data=img)
        ds.attrs["CLASS"] = np.string_("IMAGE")

        if len(img.shape) == 3 and (img.shape[0] == 3 or img.shape[2] == 3):
            ds.attrs["IMAGE_SUBCLASS"] = np.string_("IMAGE_TRUECOLOR")
            ds.attrs["IMAGE_COLORMOEL"] = np.string_("RGB")

            if img.shape[0] == 3:
                # Stored as [pixel components][height][width]
                ds.attrs["INTERLACE_MODE"] = np.string_("INTERLACE_PLANE")

            else: # This is the np standard
                # Stored as [height][width][pixel components]
                ds.attrs["INTERLACE_MODE"] = np.string_("INTERLACE_PIXEL")

        else:
            ds.attrs["IMAGE_SUBCLASS"] = np.string_("IMAGE_GRAYSCALE")
            ds.attrs["IMAGE_WHITE_IS_ZERO"] = np.array(0, dtype="uint8")
            ds.attrs["IMAGE_MINMAXRANGE"] = [img.min(), img.max()]

        ds.attrs["DISPLAY_ORIGIN"] = np.string_("UL") # not rotated
        ds.attrs["IMAGE_VERSION"] = np.string_("1.2")




# if __name__ == "__main__":
    # path = './CER6_H20_DG3d0000.ibw'
    # test = IBW2HDF5(path)
    # test.close()
