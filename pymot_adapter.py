# vim: expandtab:ts=4:sw=4
import argparse
import numpy as np
import deep_sort

import pymotutils
from pymotutils.contrib.datasets import kitti


class DeepSortTracker(pymotutils.Tracker):

    def __init__(
            self, max_cosine_distance=0.2, nn_budget=100, max_iou_distance=0.7,
            max_age=30, n_init=3, cnn_model_file=None):

        def init_fn(start_idx):
            self._metric = deep_sort.NearestNeighborDistanceMetric(
                "cosine", max_cosine_distance, nn_budget)
            self.tracker = deep_sort.Tracker(
                self._metric, max_iou_distance, max_age, n_init)
            self.trajectories = {}
            self._frame_idx = start_idx

        self._encoder = deep_sort.create_box_encoder(
            model_filename=cnn_model_file)
        self._init_fn = init_fn
        self._metric = None
        self.tracker = None
        self.trajectories = None
        self._frame_idx = 0

    def reset(self, start_idx, end_idx):
        del end_idx  # Unused variable
        self._init_fn(start_idx)

    def process_frame(self, frame_data):
        bgr_image = frame_data["bgr_image"]
        rois = [d.roi for d in frame_data["detections"]]
        features = self._encoder(bgr_image, rois)
        confidences = [d.confidence for d in frame_data["detections"]]

        deep_sort_detections = [
            deep_sort.Detection(rois[i], confidences[i], features[i])
            for i in range(len(rois))]

        self.tracker.predict()
        self.tracker.update(deep_sort_detections)

        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            roi = track.to_tlwh()
            self.trajectories.setdefault(track.track_id, []).append(
                pymotutils.Detection(
                    frame_idx=self._frame_idx, sensor_data=roi))

        self._frame_idx += 1

    def compute_trajectories(self):
        track_ids = sorted(self.trajectories)
        return [self.trajectories[tid] for tid in track_ids]


def draw_online_tracking_results(image_viewer, frame_data, deep_sort_adapter):
    del frame_data  # Unused variable.
    image_viewer.thickness = 2
    for track in deep_sort_adapter.tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            image_viewer.color = 125, 125, 125
            image_viewer.gaussian(
                track.mean[:2], track.covariance[:2, :2],
                label="%d" % track.track_id)
        else:
            image_viewer.color = pymotutils.create_unique_color_uchar(
                track.track_id)
            image_viewer.rectangle(
                *track.to_tlwh().astype(np.int), label=str(track.track_id))


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Min-cost flow tracking")
    parser.add_argument(
        "--kitti_dir", help="Path to KITTI training/testing directory",
        required=True)
    parser.add_argument(
        "--sequence", help="A four digit sequence number", required=True)
    parser.add_argument(
        "--cnn_model", default="resources/networks/mars-small128.ckpt-68577",
        help="Path to CNN checkpoint file")
    parser.add_argument(
        "--min_confidence", help="Detector confidence threshold", type=float,
        default=-1.0)
    parser.add_argument(
        "--max_num_misses",
        help="The maximum number of consecutive misses on each individual "
        "object trajectory.", type=int, default=5)
    return parser.parse_args()


def main():
    """Main program entry point."""
    args = parse_args()

    devkit = kitti.Devkit(args.kitti_dir)
    data_source = devkit.create_data_source(
        args.sequence, kitti.OBJECT_CLASSES_PEDESTRIANS)

    data_source.detections = pymotutils.preprocessing.filter_detections(
        data_source.detections, min_confidence=0., min_height=25.)

    tracker = DeepSortTracker(cnn_model_file=args.cnn_model)
    visualization = pymotutils.MonoVisualization(
        update_ms=kitti.CAMERA_UPDATE_IN_MS,
        window_shape=kitti.CAMERA_IMAGE_SHAPE,
        online_tracking_visualization=draw_online_tracking_results)
    application = pymotutils.Application(data_source)

    application.process_data(tracker, visualization)
    application.compute_trajectories(interpolation=False)
    application.play_hypotheses(visualization)


if __name__ == "__main__":
    main()
