import logging
import numpy as np
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    HallucinationMetric,
    BiasMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase
from llama_index.core import PromptTemplate
from llama_index.core.schema import NodeWithScore, TextNode
from uptrain import Settings, Evals, EvalLlamaIndex, operators
from uptrain.framework import DataSchema

from jury import Jury
import jury
import evaluate
from llama_index.core.evaluation import FaithfulnessEvaluator, CorrectnessEvaluator, GuidelineEvaluator
from llama_index.core.evaluation import BaseEvaluator, EvaluationResult
from llama_index.core.evaluation import AnswerRelevancyEvaluator, RelevancyEvaluator, SemanticSimilarityEvaluator
import typing as t
import polars as pl
from uptrain.framework.evals import ParametricEval, ResponseMatching
from uptrain.framework.evalllm import EvalLLM

import nest_asyncio

ppl_bug_number = 0

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
class EvaluationResult:
    def __init__(self, metrics = None):
        self.results = {
            "n":0,
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
        evaluation_metrics = ["Llama_retrieval_Faithfulness", "Llama_retrieval_Relevancy", "Llama_response_correctness",
                              "Llama_response_semanticSimilarity", "Llama_response_answerRelevancy","Llama_retrieval_RelevancyG",
                              "Llama_retrieval_FaithfulnessG",
                              "DeepEval_retrieval_contextualPrecision","DeepEval_retrieval_contextualRecall",
                              "DeepEval_retrieval_contextualRelevancy","DeepEval_retrieval_faithfulness",
                              "DeepEval_response_answerRelevancy","DeepEval_response_hallucination",
                              "DeepEval_response_bias","DeepEval_response_toxicity",
                              "UpTrain_Response_Completeness","UpTrain_Response_Conciseness","UpTrain_Response_Relevance",
                              "UpTrain_Response_Valid","UpTrain_Response_Consistency","UpTrain_Response_Response_Matching",
                              "UpTrain_Retrieval_Context_Relevance","UpTrain_Retrieval_Context_Utilization",
                              "UpTrain_Retrieval_Factual_Accuracy","UpTrain_Retrieval_Context_Conciseness",
                              "UpTrain_Retrieval_Code_Hallucination",
                              "NLG_chrf", "NLG_meteor", "NLG_rouge", "NLG_wer", "NLG_cer", "NLG_chrf_pp",
                             "NLG_perplexity",
                               "NLG_rouge_rouge1", "NLG_rouge_rouge2", "NLG_rouge_rougeL", "NLG_rouge_rougeLsum"
                              ]
        self.evaluationName = evaluation_metrics
        for i in Map_Uptrain_metrics_truth_val.keys():
            evaluation_metrics.append(i)
        evaluation_metrics_rev = []
        for i in evaluation_metrics:
            evaluation_metrics_rev.append(i+"_rev")
        self.metrics_results = {}
        for metric in evaluation_metrics:
            self.metrics_results[metric] = {"score": 0, "count": 0}
            self.metrics_results[metric+"_rev"] = {"score": 0, "count": 0}

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
                    print(f"{key}: {value/self.results['n']}")
        for key, value in self.metrics_results.items():
            if key in self.metrics:
                if value['count'] == 0:
                    print(f"{key}: 0, valid number : {value['count']}")
                else:
                    print(f"{key}: {value['score']/value['count']}, valid number : {value['count']}")

    def get_results_str(self):
        ans = ''
        cur = 0
        for key, value in self.results.items():
            if key in self.metrics:
                if key == 'n':
                    ans += f"{key}: {value}"+ ",\t"
                else:
                    ans += f"{key}: {value/self.results['n']:.2f}"+ ",\t"
                cur += 1
                if cur % 5 == 0:
                    ans += "\n"
        for key, value in self.metrics_results.items():
            if key in self.metrics:
                if value['count'] == 0:
                    ans += f"{key}: 0, valid number : {value['count']}"+",\t"
                else:
                    ans += f"{key}: {value['score']/value['count']:.2f}, valid number : {value['count']}"+",\t"
                cur += 1
                if cur % 5 == 0:
                    ans += "\n"
        return ans


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
                    f.writelines(f"{key}: {value/self.results['n']}\n")
        for key, value in self.metrics_results.items():
            if key in self.metrics:
                if value['count'] == 0:
                    f.writelines(f"{key}: 0, valid number : {value['count']}\n")
                else:
                    f.writelines(f"{key}: {value['score']/value['count']}, valid number : {value['count']}\n")

# 在使用uptrain本地模型前，请下载ollama
def upTrain_evaluate_self(
    settings,
    data: t.Union[list[dict], pl.DataFrame],
    checks: list[t.Union[str, Evals, ParametricEval]],
    project_name: str = None,
    schema: t.Union[DataSchema, dict[str, str], None] = None,
    metadata: t.Optional[dict[str, str]] = None,
):
    client = EvalLLM(settings)

    nest_asyncio.apply()

    results = client.evaluate(
        data=data, checks=checks, schema=schema, metadata=metadata
    )
    return results
# 咱们定义的指标对应Uptrain中的指标
Map_Uptrain_metrics_truth_val = {
    "UpTrain_Response_Completeness": Evals.RESPONSE_COMPLETENESS,
    "UpTrain_Response_Conciseness": Evals.RESPONSE_CONCISENESS,
    "UpTrain_Response_Relevance": Evals.RESPONSE_RELEVANCE,
    "UpTrain_Response_Valid": Evals.VALID_RESPONSE,
    "UpTrain_Response_Consistency": Evals.RESPONSE_CONSISTENCY,
    "UpTrain_Response_Response_Matching":ResponseMatching(method = 'llm'),

    "UpTrain_Retrieval_Context_Relevance":Evals.CONTEXT_RELEVANCE,
    "UpTrain_Retrieval_Context_Utilization":Evals.RESPONSE_COMPLETENESS_WRT_CONTEXT,
    "UpTrain_Retrieval_Factual_Accuracy":Evals.FACTUAL_ACCURACY,
    "UpTrain_Retrieval_Context_Conciseness":Evals.CONTEXT_CONCISENESS,
    "UpTrain_Retrieval_Code_Hallucination":Evals.CODE_HALLUCINATION,

}
# 指标对应得分的名字
Map_Uptrain_metrics_score_name = {
    "UpTrain_Response_Completeness":        "score_response_completeness",
    "UpTrain_Response_Conciseness":         "score_response_conciseness",
    "UpTrain_Response_Relevance":           "score_response_relevance",
    "UpTrain_Response_Valid":               "score_valid_response",
    "UpTrain_Response_Consistency":         "score_response_consistency",
    "UpTrain_Response_Response_Matching":   "score_response_match_recall",

    "UpTrain_Retrieval_Context_Relevance":  "score_context_relevance",
    "UpTrain_Retrieval_Context_Utilization":"score_response_completeness_wrt_context",
    "UpTrain_Retrieval_Factual_Accuracy":   "score_factual_accuracy",
    "UpTrain_Retrieval_Context_Conciseness":"score_context_conciseness",
    "UpTrain_Retrieval_Code_Hallucination": "score_code_hallucination",
}

#

NLG_EVALUATION_METRICS = [
    "chrf", "meteor", "wer", "cer", "chrf_pp", "perplexity",
    "rouge_rouge1", "rouge_rouge2", "rouge_rougeL", "rouge_rougeLsum"
]
#

def NLGEvaluate(questions, actual_responses, expect_answers, golden_context_ids, metrics):
    # omit_metrics = []
    # for metric in NLG_EVALUATION_METRICS:
    #     if metric not in metrics:
    #         omit_metrics.append(metric)


    # n = NLGEval(metrics_to_omit=omit_metrics)
    references = []
    if type(expect_answers) == list:
        references = [str(response) for response in expect_answers]
    elif type(expect_answers) == str:
        references = [expect_answers]

    if type(actual_responses) == list:
        predictions = [str(response) for response in actual_responses]
    elif type(actual_responses) == str:
        predictions = [actual_responses]

    # Individual Metrics
    # scores = n.compute_individual_metrics(ref=reference, hyp=hypothesis)
    scorer = Jury(metrics=["chrf", "meteor", "rouge", "wer", "cer"])
    scores = {}
    # chrf++
    chrf_plus = evaluate.load("chrf")
    score = chrf_plus.compute(predictions=predictions, references=[references], word_order=2)
    scores["chrf_pp"] = score["score"] / 100
    # perplexity:model id are needed
    perplexity = jury.load_metric("perplexity")
    score = perplexity.compute(predictions=predictions, references=references, model_id="openai-community/gpt2")
    scores["perplexity"] = score["mean_perplexity"]
    if int(scores["perplexity"]) > 1600:
        global ppl_bug_number
        ppl_bug_number = ppl_bug_number + 1
        print("\n\n" + "ppl_bug_number:" + str(ppl_bug_number) + "\n\n")
        scores["perplexity"] = 0
    #
    score = scorer(predictions=predictions, references=[references])
    scores["chrf"] = score["chrf"]["score"]
    scores["meteor"] = score["meteor"]["score"]
    #'rouge1': 0.6666666666666665, 'rouge2': 0.5714285714285715, 'rougeL': 0.6666666666666665, 'rougeLsum': 0.6666666666666665
    scores["rouge_rouge1"] = score["rouge"]["rouge1"]
    scores["rouge_rouge2"] = score["rouge"]["rouge2"]
    scores["rouge_rougeL"] = score["rouge"]["rougeL"]
    scores["rouge_rougeLsum"] = score["rouge"]["rougeLsum"]
    scores["wer"] = score["wer"]["score"]
    scores["cer"] = score["cer"]["score"]
    return scores

def UptrainEvaluate(evalModelAgent,question, actual_response, retrieval_context, expected_answer, gold_context, checks, local_model="qwen:7b-chat-v1.5-q8_0"):
    data = list()
    # 这块我把Uptrain里面的评测函数提取出来了，retrieval_context是拼接成字符串作为参数的
    retrieval_context_str = "\n".join(
        [c for c in retrieval_context]
    )
    golden_context_str = "\n".join(
        [c for c in gold_context]
    )
    data.append({"question":question,
                 "response":actual_response,
                 "context":retrieval_context_str,
                 "concise_context":golden_context_str,
                 "ground_truth":expected_answer
                 })
    # 指标所需要的变量会在实现中自己取得
    # settings是使用model的设置， data是指标所需的数据， checks 是需要检测的指标的list
    results = upTrain_evaluate_self(settings=evalModelAgent.uptrainSetting, data=data, checks=checks)
    return results
def get_DeepEval_Metrices(evalModelAgent,model_name="DeepEval_retrieval_contextualPrecision"):
    match model_name:
        case "DeepEval_retrieval_contextualPrecision":
            return ContextualPrecisionMetric(threshold=0.7,model=evalModelAgent.deepEvalModel,include_reason=True)
        case "DeepEval_retrieval_contextualRecall":
            return ContextualRecallMetric(threshold=0.7,model=evalModelAgent.deepEvalModel,include_reason=True)
        case "DeepEval_retrieval_contextualRelevancy":
            return ContextualRelevancyMetric(threshold=0.7,model=evalModelAgent.deepEvalModel,include_reason=True)
        case "DeepEval_retrieval_faithfulness":
            return FaithfulnessMetric(threshold=0.7,model=evalModelAgent.deepEvalModel,include_reason=True)
        case "DeepEval_response_answerRelevancy":
            return AnswerRelevancyMetric(threshold=0.7,model=evalModelAgent.deepEvalModel,include_reason=True)
        case "DeepEval_response_hallucination":
            return HallucinationMetric(threshold=0.7,model=evalModelAgent.deepEvalModel,include_reason=True)
        case "DeepEval_response_bias":
            return BiasMetric(threshold=0.7,model=evalModelAgent.deepEvalModel,include_reason=True)
        case "DeepEval_response_toxicity":
            return ToxicityMetric(threshold=0.7,model=evalModelAgent.deepEvalModel,include_reason=True)
# todo: ours evaluator
# class Evaluator(BaseEvaluator):
def get_llama_evaluator(evalModelAgent,model_name="Llama_retrieval_Faithfulness"):
    match model_name:
        case "Llama_retrieval_Faithfulness":
            return FaithfulnessEvaluator(llm=evalModelAgent.llamaModel)
        case "Llama_retrieval_FaithfulnessG":
            return FaithfulnessEvaluator(llm=evalModelAgent.llamaModel, eval_template=LLAMA_CUSTOM_FAITHFULNESS_TEMPLATE)
        case "Llama_retrieval_Relevancy":
            return RelevancyEvaluator(llm=evalModelAgent.llamaModel)
        case "Llama_retrieval_RelevancyG":
            return RelevancyEvaluator(llm=evalModelAgent.llamaModel)
        case "Llama_response_correctness":
            return CorrectnessEvaluator(llm=evalModelAgent.llamaModel)
        case "Llama_response_semanticSimilarity":
            return SemanticSimilarityEvaluator()
        case "Llama_response_answerRelevancy":
            return AnswerRelevancyEvaluator(llm=evalModelAgent.llamaModel)

# response evaluate
def evaluating(question, response, actual_response, retrieval_context, retrieval_ids, expected_answer, golden_context, golden_context_ids, metrics, evalModelAgent):

    # 创建一个新类，主要是用来记录各个指标有效的个数以及得分
    eval_result = EvaluationResult()
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
    upTrain_metrics = list()
    # region llama_index evaluation
    for i in eval_result.evaluationName:
        if i in metrics and i[0:8] != "DeepEval" and i[0:7] != "UpTrain" and i[0:3] != "NLG":
            print("now run " + i)

            count = 2
            while True:
                try:
                    evaluator = get_llama_evaluator(evalModelAgent, i)
                    res = EvaluationResult()
                    res.passing = False
                    if i == "Llama_retrieval_FaithfulnessG":
                        # 这块跟prompt相关（LLAMA_CUSTOM_FAITHFULNESS_TEMPLATE）
                        query_str = f"Question: {question}\nInformation: {response.response}"
                        # Faithfulness.evaluate_response的实现中会把query的值传给上面prompt的query_str
                        # 实现的功能和文档中相同，这里试用了一下gpt生成的prompt
                        res = evaluator.evaluate_response(query=query_str, response=response)
                    elif i == "Llama_retrieval_RelevancyG":
                        # 这块要评测golden_context和其他的关系，相当于在原基础上更换retrieval_context
                        # 在Relevancy的实现中，会取出node里面的文本，所以我们构建的时候只传文本就好
                        # 这块语义差不多所以就没更换prompt
                        response.source_nodes.clear()
                        for context in golden_context:
                            node = TextNode()
                            node.text = context
                            temp = NodeWithScore(node=node)
                            response.source_nodes.append(temp)
                        res = evaluator.evaluate_response(query=question, response=response)
                    else:
                        # 这里的传参主要是想要使用系统自带的函数，在实现中会自动提取所需的内容
                        res = evaluator.evaluate_response(query=question, response=response, reference=expected_answer)
                    if res.passing:
                        eval_result.metrics_results[i]["score"] = 1
                    elif 0 <= res.score and res.score <= 1:
                        eval_result.metrics_results[i]["score"] = res.score
                    # 这里要记录一下数量，因为由于种种原因这个过程会报错，就会导致无效的记录
                    eval_result.metrics_results[i]["count"] = 1
                    break
                except Exception as e:
                    count = count - 1

                    logging.exception(e)
                    print("error ")
                    if count == 0:
                        print(i + " error")
                        break
    # endregion
    # region uptrain evaluation
    for i in metrics:
        if i in Map_Uptrain_metrics_truth_val.keys():
            upTrain_metrics.append(i)
    if upTrain_metrics.__len__() != 0:
        # upTrain 可以进行多个指标的评测（传入要评测指标的list）,因此我们将要测的指标封装成一个list
        upTrain_metrics_val = list()
        for i in upTrain_metrics:
            upTrain_metrics_val.append(Map_Uptrain_metrics_truth_val[i])
        count = 2
        while True:
            try:
                result = UptrainEvaluate(evalModelAgent, question,actual_response,retrieval_context,expected_answer,golden_context,upTrain_metrics_val)
                for i in upTrain_metrics:
                    try:
                        #TODO: fix NONETYPE
                        if result[0][Map_Uptrain_metrics_score_name[i]] >= 0 and result[0][Map_Uptrain_metrics_score_name[i]] <= 1:
                            eval_result.metrics_results[i]["score"] = result[0][Map_Uptrain_metrics_score_name[i]]
                            eval_result.metrics_results[i]["count"] = 1
                    except Exception as e:
                        logging.exception(e)
                break
            except Exception as e:
                count = count - 1
                logging.exception(e)
                if count == 0:
                    break
    # endregion
    # region deepEval evaluation
    for i in eval_result.evaluationName:
        if i in metrics and i[0:8] == "DeepEval":
            count = 2
            while True:
                try:
                    # 构建评测的参数
                    test_case = LLMTestCase(
                        input=question,
                        actual_output=actual_response,
                        retrieval_context=retrieval_context,
                        expected_output=expected_answer,
                        context=golden_context,
                    )
                    deepeval_metric = get_DeepEval_Metrices(evalModelAgent, i)
                    deepeval_metric.measure(test_case)
                    if len(deepeval_metric.verdicts) == 0:
                        raise Exception("deepeval verdicts is zero")

                    if deepeval_metric.score:
                        eval_result.metrics_results[i]["score"] = 1
                    eval_result.metrics_results[i]["count"] = 1
                    break
                except Exception as e:
                    count = count - 1

                    logging.exception(e)
                    #   file2.write(traceback.format_exc())
                    print("error ")
                    if count == 0:
                        print(i + " error")
                        break
    # region NLG evaluation
    NLG_metrics = []
    for i in metrics:
        if i[0:3] == "NLG":
            NLG_metrics.append(i[4:])
    if NLG_metrics.__len__() != 0:
        result = NLGEvaluate(question, actual_response, expected_answer, golden_context_ids, NLG_metrics)
        for i in NLG_EVALUATION_METRICS:
            eval_result.metrics_results["NLG_"+i]["score"] = result[i]
            eval_result.metrics_results["NLG_"+i]["count"] = 1

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
    score=1.0 if is_hit else 0.0
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
