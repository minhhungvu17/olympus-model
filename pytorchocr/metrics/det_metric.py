from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__all__ = ["DetMetric", "DetFCEMetric"]

from .eval_det_iou import DetectionIoUEvaluator


class DetMetric(object):
    def __init__(self, main_indicator="hmean") -> None:
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch):
        """_summary_

        Args:
            preds (_type_): _description_
            batch (_type_): _description_
        """
        gt_polygons_batch = batch[2]
        ignore_tags_batch = batch[3]
        for pred, gt_polygons, ignore_tags in zip(
            preds, gt_polygons_batch, ignore_tags_batch
        ):
            # prepare gt
            gt_info_list = [
                {"points": gt_polygon, "text": "", "ignore": ignore_tag}
                for gt_polygon, ignore_tag in zip(gt_polygons, ignore_tags)
            ]

            # prepare det
            det_info_list = [
                {"points": det_polygon, "text": ""} for det_polygon in pred["points"]
            ]

            result = self.evaluator.evaluate_image(gt_info_list, det_info_list)
            self.results.append(result)

    def get_metric(self):
        """Return metrics {
            'precision': 0,
            'recall': 0,
            'hmean': 0
        }
        """
        metrics = self.evaluator.combine_results(self.results)
        self.reset()
        return metrics

    def reset(self):
        self.results = []  # clear results


class DetFCEMetric(object):
    def __init__(self, main_indicator="hmean") -> None:
        self.evaluator = DetectionIoUEvaluator()
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, preds, batch):
        gt_polygons_batch = batch[2]
        ignore_tags_batch = batch[3]

        for pred, gt_polygons, ignore_tags in zip(
            preds, gt_polygons_batch, ignore_tags_batch
        ):
            # prepare gt
            gt_info_list = [
                {"points": gt_polygon, "text": "", "ignore": ignore_tag}
                for gt_polygon, ignore_tag in zip(gt_polygons, ignore_tags)
            ]
            # prepare det
            det_info_list = [
                {"points": det_polygon, "text": "", "score": score}
                for det_polygon, score in zip(pred["points"], pred["scores"])
            ]
            for score_th in self.results.keys():
                det_info_list_thr = [
                    det_info
                    for det_info in det_info_list
                    if det_info["score"] >= score_th
                ]
                result = self.evaluator.evaluate_image(gt_info_list, det_info_list_thr)
                self.results[score_th].append(result)

    def get_metric(self):
        metrics = {}
        hmean = 0
        for score_th in self.results.keys():
            metric = self.evaluator.combine_results(self.results[score_th])
            metric_str = "precision:{:.5f} recall:{:.5f} hmean:{:.5f}".format(
                metric["precision"], metric["recall", metric["hmean"]]
            )
            hmean = max(hmean, metric["hmean"])
        metrics["hmean"] = hmean
        self.reset()
        return metrics

    def reset(self):
        self.results = {0.3: [], 0.4: [], 0.5: [], 0.6: [], 0.7: [], 0.8: [], 0.9: []}
