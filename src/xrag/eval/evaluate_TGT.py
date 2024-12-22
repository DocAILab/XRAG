from jury import Jury
import jury
import evaluate

NLG_EVALUATION_METRICS = [
    "chrf", "meteor", "wer", "cer", "chrf_pp", "mauve", "perplexity",
    "rouge_rouge1", "rouge_rouge2", "rouge_rougeL", "rouge_rougeLsum"
]


def NLGEvaluate(questions, actual_responses, golden_contexts, golden_context_ids, metrics):
    # omit_metrics = []
    # for metric in NLG_EVALUATION_METRICS:
    #     if metric not in metrics:
    #         omit_metrics.append(metric)

    # n = NLGEval(metrics_to_omit=omit_metrics)
    references = []
    for context in golden_contexts:
        references.append(str(context))
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
    scores["chrf_pp"] = score["score"]
    # MAUVE 一个 predictions 只能对应一个references
    mauve = evaluate.load('mauve')
    score = mauve.compute(predictions=predictions, references=[''.join(references)])
    scores["mauve"] = score.mauve
    # perplexity:model id are needed
    perplexity = jury.load_metric("perplexity")
    score = perplexity.compute(predictions=predictions, references=references, model_id="openai-community/gpt2")
    scores["perplexity"] = score["mean_perplexity"]
    #
    score = scorer(predictions=predictions, references=[references])
    scores["chrf"] = score["chrf"]["score"]
    scores["meteor"] = score["meteor"]["score"]
    # 'rouge1': 0.6666666666666665, 'rouge2': 0.5714285714285715, 'rougeL': 0.6666666666666665, 'rougeLsum': 0.6666666666666665
    scores["rouge_rouge1"] = score["rouge"]["rouge1"]
    scores["rouge_rouge2"] = score["rouge"]["rouge2"]
    scores["rouge_rougeL"] = score["rouge"]["rougeL"]
    scores["rouge_rougeLsum"] = score["rouge"]["rougeLsum"]
    scores["wer"] = score["wer"]["score"]
    scores["cer"] = score["cer"]["score"]
    return scores
