import json
import os
import os.path as osp
import time

import numpy as np


def convert_json(obj):
    """ Convert obj to a version which can be serialized with JSON. """
    if is_json_serializable(obj):
        return obj
    else:
        if isinstance(obj, dict):
            return {convert_json(k): convert_json(v)
                    for k, v in obj.items()}
        
        elif isinstance(obj, tuple):
            return (convert_json(x) for x in obj)
        
        elif isinstance(obj, list):
            return [convert_json(x) for x in obj]
        
        elif hasattr(obj, '__name__') and not ('lambda' in obj.__name__):
            return convert_json(obj.__name__)
        
        elif hasattr(obj, '__dict__') and obj.__dict__:
            obj_dict = {convert_json(k): convert_json(v)
                        for k, v in obj.__dict__.items()}
            return {str(obj): obj_dict}
        
        return str(obj)


def is_json_serializable(v):
    try:
        json.dumps(v)
        return True
    except:
        return False


def get_exp_name(env_name):
    exp_name = str(env_name) + '-' + '-'.join([x.replace(':', '-') for x in time.ctime().split()[2:4]])
    return exp_name


def proc_id():
    try:
        from mpi4py import MPI
    except ImportError:
        return 0
    return MPI.COMM_WORLD.Get_rank()


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def dump_config(config, exp_name, output_dir, ):
    config_json = convert_json(config)
    if exp_name is not None:
        config_json['exp_name'] = exp_name
    if proc_id() == 0:
        output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=False)
        print(colorize('Saving config:\n', color='cyan', bold=True))
        print(output)
        with open(osp.join(output_dir, "config.json"), 'w') as out:
            out.write(output)


def log_config(config, output_dir):
    config_json = convert_json(config)
    if proc_id() == 0:
        output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=False)
        print(colorize('Saving config:\n', color='cyan', bold=True))
        print(output)
        with open(osp.join(output_dir, "config.json"), 'w') as out:
            out.write(output)


def statistics_scalar(x):
    x = np.array(x, dtype=np.float32)
    mean = np.sum(x) / len(x)
    std = np.sqrt(np.sum((x - mean) ** 2))
    min_val = np.min(x) if len(x) > 0 else np.inf
    max_val = np.max(x) if len(x) > 0 else -np.inf
    return mean, std, min_val, max_val


class Monitor:
    def __init__(self):
        self.epoch_dict = dict()
    
    def store(self, **kwargs):
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            if type(v) == list:
                self.epoch_dict[k].extend(v)
            else:
                self.epoch_dict[k].append(v)
    
    def log(self, key):
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
        stats = statistics_scalar(vals)
        self.epoch_dict[key] = []
        return {
            'mean': stats[0],
            'std': stats[1],
            'min_val': stats[2],
            'max_val': stats[3],
        }


class Timer:
    def __init__(self, stdout=False):
        self._start_times = dict()
        self.timing_dict = dict()
        self._stdout= stdout
    
    def clear(self):
        self._start_times = dict()
        self.timing_dict = dict()
    
    def start(self, name):
        self._start_times[name] = self.current_time
        if self._stdout:
            print('Staring', name, '...')
    
    def end(self, name):
        assert name in self._start_times
        self.timing_dict[name] = self.current_time - self._start_times[name]
        if self._stdout:
            print('Ending', name, '...')
    
    def get_time(self, name):
        assert name in self.timing_dict
        return self.timing_dict[name]
    
    @property
    def current_time(self):
        return time.time()


def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_video(paths, filename):
    import cv2
    assert all(['ims' in path for path in paths])
    ims = [im for path in paths for im in path['ims']]
    _make_dir(filename)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = 30.0
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()


def merge_configs(list_of_configs):
    master_config = dict()
    for c in list_of_configs:
        master_config.update(c)
    return master_config
