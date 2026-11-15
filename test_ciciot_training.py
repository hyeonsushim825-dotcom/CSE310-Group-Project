"""CPU-only integration tests; real project datasets and a GPU are not needed.

Run: python -m unittest -v test_ciciot_training
All generated data/checkpoints stay in an automatically removed temp directory.
"""
import csv
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from train_ciciot import load_features, load_trained_model, training_centroid, anomaly_scores, sha256
from check_ciciot_data import check_label_alignment


ROOT = Path(__file__).resolve().parent


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory(prefix='clan-tests-')
        cls.folder = Path(cls.tmp.name)
        cls.x = np.random.default_rng(12).normal(size=(67, 39)).astype('float32')
        # Deliberately include outliers: training must retain them without labels.
        cls.x[-7:] += 8
        cls.source = cls.folder / 'features.npy'
        np.save(cls.source, cls.x)
        cls.source_hash = sha256(cls.source)
        cls.run_train('full')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def run_train(cls, name, *extra, success=True):
        command = [sys.executable, str(ROOT / 'train_ciciot.py'),
                   '--train-npy', str(cls.source), '--run-dir', str(cls.folder / name),
                   '--device', 'cpu', '--epochs', '3', '--batch-size', '16',
                   '--neurons', '16,16', '--seed', '42', *extra]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120,
                                env={**os.environ, 'OMP_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'})
        if success and result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        if not success and not result.returncode:
            raise AssertionError('Expected rejection, but command succeeded')
        return result

    def test_features_and_invalid_inputs(self):
        self.assertTrue(np.array_equal(load_features(self.source), self.x))
        for index, x in enumerate([np.zeros(67), np.zeros((67,72)),
                                   np.full((4,39), np.nan), np.full((4,39), 1e300),
                                   np.ones((4,39), dtype=complex)]):
            path = self.folder / f'invalid_{index}.npy'
            np.save(path, x)
            with self.assertRaises(ValueError):
                load_features(path)

    def test_checkpoint_all_rows_and_scores(self):
        model, checkpoint = load_trained_model(self.folder / 'full/final.pt')
        self.assertEqual(checkpoint['metadata']['used_shape'], [67,39])
        self.assertFalse(checkpoint['metadata']['training_labels_used'])
        self.assertEqual(checkpoint['model_config']['n_classes'], None)
        self.assertEqual(checkpoint['epoch'], 3)
        self.assertEqual(sha256(self.source), self.source_hash)
        with torch.no_grad():
            embeddings = model(torch.from_numpy(self.x))
        self.assertEqual(tuple(embeddings.shape), (67,64))
        self.assertTrue(torch.allclose(checkpoint['centroid'], embeddings.double().mean(0).float(), atol=1e-6))
        self.assertTrue(torch.allclose(checkpoint['centroid'], training_centroid(model,self.x,'cpu',7), atol=1e-6))
        reference = 1 - torch.nn.functional.cosine_similarity(embeddings,checkpoint['centroid'][None],dim=1)
        # Uneven batches check row order and final singleton handling.
        actual = anomaly_scores(model,self.x,checkpoint['centroid'],'cpu',batch_size=11)
        np.testing.assert_allclose(actual, reference.numpy(), atol=1e-6)

    def test_reproducibility_and_resume(self):
        self.run_train('repeat')
        self.run_train('paused', '--stop-after-epoch', '1')
        self.assertFalse((self.folder / 'paused/final.pt').exists())
        self.run_train('paused', '--resume')
        _, full = load_trained_model(self.folder / 'full/final.pt')
        for name in ['repeat', 'paused']:
            _, actual = load_trained_model(self.folder / name / 'final.pt')
            for key in full['model_state_dict']:
                self.assertTrue(torch.equal(full['model_state_dict'][key], actual['model_state_dict'][key]), key)
            self.assertTrue(torch.equal(full['centroid'], actual['centroid']))
            self.assertEqual([r['loss'] for r in full['history']], [r['loss'] for r in actual['history']])
            with (self.folder / name / 'loss.csv').open() as handle:
                self.assertEqual([int(row['epoch']) for row in csv.DictReader(handle)], [1,2,3])

    def test_overwrite_and_resume_guards(self):
        result = self.run_train('full', success=False)
        self.assertIn('Refusing to overwrite', result.stderr)
        self.run_train('guard', '--stop-after-epoch', '1')
        result = self.run_train('guard','--resume','--epochs','4',success=False)
        self.assertIn('Resume mismatch', result.stderr)
        alternate = self.folder / 'altered.npy'
        np.save(alternate, self.x + 1)
        result = self.run_train('guard','--resume','--train-npy',str(alternate),success=False)
        self.assertIn('data_sha256', result.stderr)

    def test_label_quality_checks(self):
        clean = self.x[:60]
        labels = np.r_[np.zeros(60,dtype=np.uint8), np.ones(7,dtype=np.uint8)]
        result = check_label_alignment(self.x, labels, clean, 7)
        self.assertEqual(result['benign_rows'],60)
        permutation = np.random.default_rng(10).permutation(len(self.x))
        check_label_alignment(self.x[permutation], labels[permutation], clean, 7)
        with self.assertRaises(ValueError):
            check_label_alignment(self.x,np.roll(labels,1),clean,7)

    def test_default_network_one_epoch(self):
        self.run_train('default_network','--neurons','1024,1024,1024,1024',
                       '--epochs','1','--max-samples','16','--batch-size','8')
        _, checkpoint = load_trained_model(self.folder / 'default_network/final.pt')
        self.assertEqual(checkpoint['model_config']['neurons'],[1024]*4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
