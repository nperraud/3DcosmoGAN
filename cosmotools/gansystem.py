import numpy as np
import tensorflow as tf

from gantools.gansystem import UpscaleGANsystem
from .model import cosmo_metric_list
import itertools


class CosmoUpscaleGANsystem(UpscaleGANsystem):
    def default_params(self):


        # Global parameters
        # -----------------
        d_param = super().default_params()

        d_param['Nstats_cubes'] = 10

        return d_param


    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.params['Nstats_cubes']:
            self._cosmo_metric_list = cosmo_metric_list()
            for met in self._cosmo_metric_list:
                met.add_summary(collections="cubes")
            self._cubes_summaries = tf.summary.merge(tf.get_collection("cubes"))

    def train(self, dataset, **kwargs):
        if self.params['Nstats_cubes']:
            # Only implented for the 3dimentional case...
            assert(self.net.params['generator']['data_size']>= 2)
            assert(len(dataset._X)>=self.params['Nstats_cubes'])
            self.summary_dataset_cubes = itertools.cycle(dataset.iter_cubes(self.params['Nstats_cubes'], downscale=self.net.params['upscaling']))
            offset = next(self.summary_dataset_cubes).shape[1]//8
            self.offset = offset
            self.preprocess_summaries(dataset._X[:,offset:,offset:,offset:], rerun=False)
            self._global_score = np.inf
        super().train(dataset, **kwargs)

    def preprocess_summaries(self, X_real, **kwargs):
        if self.net.params['cosmology']['backward_map']:
            X_real = self.params['net']['cosmology']['backward_map'](X_real)
        for met in self._cosmo_metric_list:
            met.preprocess(X_real, **kwargs)

    def _train_log(self, feed_dict):
        super()._train_log(feed_dict)
        if self.params['Nstats_cubes']:
            X_real = next(self.summary_dataset_cubes)
            if self.net.params['upscaling']:
                small = X_real
                small = np.expand_dims(small, axis=self.net.params['generator']['data_size']+1)
            else:
                small = None
            X_fake = self.upscale_image(N=self.params['Nstats_cubes'],
                                        small=small,
                                        resolution=X_real.shape[1],
                                        sess=self._sess)
            offset = self.offset
            feed_dict = self.compute_summaries(X_fake[:,offset:,offset:,offset:], feed_dict)
            # m = self._cosmo_metric_list[0]
            # print(m.last_metric)
            # print(m._metrics[0].last_metric)
            # print(m._metrics[1].last_metric)
            # print(m._metrics[2].last_metric)
            new_val = self._cosmo_metric_list[0].last_metric

            if new_val <= self._global_score:
                self._global_score = new_val
                self._save(self._counter)
                print('New lower score at {}'.format(new_val))
            summary = self._sess.run(self._cubes_summaries, feed_dict=feed_dict)
            self._summary_writer.add_summary(summary, self._counter)

    def compute_summaries(self, X_fake, feed_dict={}):
        if self.net.params['cosmology']['backward_map']:
            # if X_real is not None:
            #     X_real = self.params['cosmology']['backward_map'](X_real)
            X_fake = self.net.params['cosmology']['backward_map'](X_fake)
        for met in self._cosmo_metric_list:
            feed_dict = met.compute_summary(X_fake, None, feed_dict)
        return feed_dict