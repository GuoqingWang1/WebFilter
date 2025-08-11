# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def _default_compute_score(data_source, prompt_str, solution_str, ground_truth, extra_info=None, val_type='f1', is_train = True, batch_size = 32):
    if type(data_source) != str: # SR_RR
        reslist = []
        if data_source in ['nq', "2wiki", "Bamboogle", "hotpotqa", "musique", "tq", "popqa"]:
            from . import webfilter_SR_RR_reward 
            reslist = webfilter_SR_RR_reward.compute_score_batch(solution_str, ground_truth, batch_size=batch_size)
        else:
            for data_source_e,solution_str_e,ground_truth_e in zip(data_source, solution_str, ground_truth):
                if data_source in ['nq_val', "2wiki_val", "Bamboogle_val", "hotpotqa_val", "musique_val", "tq_val", "popqa_val"]:
                    from . import format_and_f1
                    reslist = format_and_f1.compute_score(solution_str_e, ground_truth_e, val_type=val_type)
        return reslist 
    else: # SR_Only
        if data_source in ['nq', "2wiki", "Bamboogle", "hotpotqa", "musique", "tq", "popqa"]:
            from . import webfilter_SR_reward 
            res = webfilter_SR_reward.compute_score(solution_str, ground_truth, val_type=val_type)
        elif data_source in ['nq_val', "2wiki_val", "Bamboogle_val", "hotpotqa_val", "musique_val", "tq_val", "popqa_val"]: 
            from . import format_and_f1
            res = format_and_f1.compute_score(data_source=data_source, solution_str=solution_str, ground_truth=ground_truth, prompt_str=prompt_str, val_type=val_type)
        else:
            raise NotImplementedError
        if isinstance(res, (int, float, bool)):
            return float(res)
        else:
            return float(res[0])
