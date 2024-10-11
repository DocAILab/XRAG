import numpy as np
from llama_index.core import PromptTemplate

LLAMA_CUSTOM_FAITHFULNESS_TEMPLATE = PromptTemplate(
    "Please tell if the context supports the given information related to the question.\n"
    "You need to answer with either YES or NO.\n"
    "Answer YES if the context supports the information related to the question, even if most of the context is unrelated. \n"
    "See the examples below.\n"
    "Question: Is apple pie usually double-crusted?\n"
    "Information: Apple pie is generally double-crusted.\n"
    "Context: An apple pie is a fruit pie in which the principal filling ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).\n"
    "Answer: YES\n"
    "Question: Does apple pie usually taste bad?\n"
    "Information: Apple pies tastes bad.\n"
    "Context: An apple pie is a fruit pie in which the principal filling ingredient is apples. \n"
    "Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.\n"
    "It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).\n"
    "Answer: NO\n"
    "Question and Information: {query_str}\n"
    "Context: {context_str}\n"
    "Answer: "
)


class EvaluationResult_TRT:
    def __init__(self, metrics=None):
        self.results = {
            "n": 0,
            "F1": 0.0,
            "em": 0.0,
            "mrr": 0.0,
            "hit1": 0.0,
            "hit10": 0.0,
            "MAP": 0.0,
            "NDCG": 0.0,
            "DCG": 0.0,
            "IDCG": 0.0,
        }
        # 初始化用于评估的指标，每一个指标都是一个包含score和count的字典
        evaluation_metrics = []
        self.evaluationName = evaluation_metrics
        evaluation_metrics_rev = []
        for i in evaluation_metrics:
            evaluation_metrics_rev.append(i + "_rev")
        self.metrics_results = {}
        for metric in evaluation_metrics:
            self.metrics_results[metric] = {"score": 0, "count": 0}
            self.metrics_results[metric + "_rev"] = {"score": 0, "count": 0}

        if metrics is None:
            metrics = []
        metrics.append("n")
        metrics.append("F1")
        metrics.append("em")
        metrics.append("mrr")
        metrics.append("hit1")
        metrics.append("hit10")
        metrics.append("MAP")
        metrics.append("NDCG")
        metrics.append("DCG")
        metrics.append("IDCG")

        self.metrics = metrics

    def add(self, evaluate_result):
        for key in self.results.keys():
            if key in self.metrics:
                self.results[key] += evaluate_result.results[key]
        for key in self.metrics_results.keys():
            if key in self.metrics:
                if evaluate_result.metrics_results[key]["score"] != None:
                    self.metrics_results[key]["score"] += evaluate_result.metrics_results[key]["score"]
                    self.metrics_results[key]["count"] += evaluate_result.metrics_results[key]["count"]
        self.results["n"] += 1

    def print_results(self):
        for key, value in self.results.items():
            if key in self.metrics:
                if key == 'n':
                    print(f"{key}: {value}")
                else:
                    print(f"{key}: {value / self.results['n']}")
        for key, value in self.metrics_results.items():
            if key in self.metrics:
                if value['count'] == 0:
                    print(f"{key}: 0, valid number : {value['count']}")
                else:
                    print(f"{key}: {value['score'] / value['count']}, valid number : {value['count']}")

    def print_results_to_path(self, path, config, sample_arr):
        print("save data to " + path)
        f = open(path, 'a')
        f.writelines("\n")
        f.writelines("=========================== begin ==================\n")
        f.writelines("database: " + config.dataset + "\n")
        f.writelines("sample_arr: " + str(sample_arr) + "\n")
        for key, value in self.results.items():
            if key in self.metrics:
                if key == 'n':
                    f.writelines(f"{key}: {value}\n")
                else:
                    f.writelines(f"{key}: {value / self.results['n']}\n")
        for key, value in self.metrics_results.items():
            if key in self.metrics:
                if value['count'] == 0:
                    f.writelines(f"{key}: 0, valid number : {value['count']}\n")
                else:
                    f.writelines(f"{key}: {value['score'] / value['count']}, valid number : {value['count']}\n")


# response evaluate
def evaluating_TRT(retrieval_ids, golden_context_ids):
    # 创建一个新类，主要是用来记录各个指标有效的个数以及得分
    eval_result = EvaluationResult_TRT()
    # region 常规指标
    eval_result.results["F1"] = F1(retrieval_ids, golden_context_ids)
    eval_result.results["em"] = Em(retrieval_ids, golden_context_ids)
    eval_result.results["mrr"] = Mrr(retrieval_ids, golden_context_ids)
    eval_result.results["hit1"] = Hit(retrieval_ids, golden_context_ids[0:1])
    eval_result.results["hit10"] = Hit(retrieval_ids, golden_context_ids[0:10])
    eval_result.results["MAP"] = MAP(retrieval_ids, golden_context_ids)
    eval_result.results["NDCG"] = NDCG(retrieval_ids, golden_context_ids)
    eval_result.results["DCG"] = DCG(retrieval_ids, golden_context_ids)
    eval_result.results["IDCG"] = IDCG(retrieval_ids, golden_context_ids)
    # endregion
    # 由于upTrain可以一次计算多个指标，所以这个变量之后会从upTrain的多个指标

    # endregion
    return eval_result


# region commonly used indicators
def Mrr(retrieved_ids, expected_ids):
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    for i, id in enumerate(retrieved_ids):
        if id in expected_ids:
            score = 1.0 / (i + 1)
            return score
    return 0.0


def Hit(retrieved_ids, expected_ids):
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    is_hit = any(id in expected_ids for id in retrieved_ids)
    score = 1.0 if is_hit else 0.0
    return score


def F1(retrieved_ids, expected_ids):
    # 转换为集合进行交集和差集运算
    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)

    # 计算TP、FP和FN
    TP = len(retrieved_set & expected_set)  # 预测为正且实际为正的
    FP = len(retrieved_set - expected_set)  # 预测为正但实际为负的
    FN = len(expected_set - retrieved_set)  # 实际为正但预测为负的

    # 计算精确率（Precision）和召回率（Recall）
    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # 计算F1分数
    F1_score = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0

    return F1_score


def Em(retrieved_ids, expected_ids):
    retrieved_ids.sort()
    expected_ids.sort()
    if expected_ids == retrieved_ids:
        return 1
    return 0


def MAP(retrieved_ids, expected_ids):
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    if len(retrieved_ids) == 0:
        return 0.0
    score = 0.0
    for i, id in enumerate(expected_ids):
        if id in retrieved_ids:
            score += (i + 1) / (retrieved_ids.index(id) + 1)
    return score / len(expected_ids)


def DCG(retrieved_ids, expected_ids):
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    score = 0.0
    for i, id in enumerate(retrieved_ids):
        if id in expected_ids:
            # index starts at 0, so add 1 to the index
            score += (1) / np.log2((i + 1) + 1)

    return score


def IDCG(retreived_ids, expected_ids):
    temp_1 = []
    temp_2 = []
    for a in retreived_ids:
        if a in expected_ids:
            temp_1.append(a)
        else:
            temp_2.append(a)
    temp_1.append(temp_2)
    idcg = DCG(temp_1, expected_ids)
    return idcg


def NDCG(retrieved_ids, expected_ids):
    dcg = DCG(retrieved_ids, expected_ids)
    idcg = IDCG(retrieved_ids, expected_ids)
    if idcg == 0:
        return 0
    return dcg / idcg
# endregion
