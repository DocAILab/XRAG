export const messages = {
  en: {
    nav: { workflow: 'Workflow', features: 'Features', demo: 'Demo', github: 'GitHub' },
    footer: { copyright: '© 2024 XRAG. All rights reserved.', github: 'GitHub', docs: 'Docs' },
    hero: {
      title: 'XRAG: eXamining the Core - Benchmarking Foundational Components in Advanced Retrieval-Augmented Generation',
      subtitle: 'A powerful benchmarking framework for evaluating foundational components in advanced RAG systems',
      demo: 'Online Demo',
    },
    steps: ['Build Dataset', 'Build Vector Database', 'Configure LLM', 'Configure Retrieval', 'Set Evaluation Metrics', 'View Test Results'],
    common: { previous: 'Previous', next: 'Next', runEvaluation: 'Run Evaluation', cancel: 'Cancel', startOver: 'Start Over', loading: 'Loading…' },
    dataset: {
      title: 'Build Dataset', preset: 'Choose Preset Dataset', custom: 'Or Upload Custom Dataset',
      descriptions: { hotpot_qa: 'Multi-hop QA Dataset', drop: 'Numerical Reasoning Dataset', natural_questions: 'Natural Language QA Dataset' },
      folderTitle: 'Upload Document Folder', folderDesc: 'Supports PDF, TXT, DOCX, etc.', folderPath: 'Folder path', folderPlaceholder: 'Path to a folder of documents', questionsPerFile: 'Number of questions per file', sentenceLength: 'Sentence length (-1 = file-level)', outputPath: 'Output JSON path', chooseFolder: 'Generate from Folder',
      folderFeatures: ['Standard dataset support', 'Custom dataset integration', 'Folder import support'],
      jsonTitle: 'Upload JSON Dataset', jsonDesc: 'Supports custom JSON format datasets', dropJson: 'Drop or pick a JSON file', chooseJson: 'Upload JSON File', jsonFeatures: ['Supports QA pair datasets', 'Supports custom metadata', 'Supports batch import'],
      loaded: 'Loaded', documents: 'documents', questions: 'test questions', source: 'source', sources: { preset: 'preset', json: 'JSON upload', folder: 'document folder' },
    },
    vector: { title: 'Build Vector Database', embedding: 'Select Embedding Model', splitType: 'Split Type', chunkSize: 'Chunk Size', chunkOverlap: 'Chunk Overlap', windowSize: 'Sentence Window Size', chunkSizes: 'Hierarchy Chunk Sizes', persistDir: 'Persist Directory', splitTypes: {} },
    llm: { title: 'Configure LLM', select: 'Select Language Model', apiKey: 'API Key', apiBase: 'API Base URL', modelName: 'Model Name', hfModel: 'HuggingFace Model', authToken: 'Auth Token', ollamaModel: 'Ollama Model', timeout: 'Request Timeout (seconds)', temperature: 'Temperature' },
    retrieval: { title: 'Configure Retrieval', type: 'Retriever Type', retrieval: 'Retrieval', mode: 'Retriever Mode', modeValue: 'Mode', topK: 'Top K Results', pre: 'Pre-retrieval Process', noneProcess: 'No Processing', rerank: 'Post-process Rerank', noneRerank: 'No Reranking', orchestrator: 'Orchestrator', synthesizer: 'Response Synthesizer', qaTemplate: 'QA Template', refineTemplate: 'Refine Template' },
    metrics: { title: 'Set Evaluation Metrics', samples: 'Number of test samples', experiment: 'Run in experiment-1 mode (use cfg.test_init_total_number_documents instead of cfg.n)', groups: {} },
    results: { title: 'View Test Results', experimentId: 'Experiment ID:', running: 'Running evaluation', complete: 'samples complete.', failed: 'Evaluation failed:', cancelled: 'Evaluation cancelled.', finished: 'Evaluation complete', processed: 'samples processed.', aggregate: 'Aggregated Metrics', metric: 'Metric', score: 'Score', valid: 'Valid Count', perSample: 'Per-sample results', answer: 'Answer:', expected: 'Expected:', retrieved: 'Retrieved', chunks: 'context chunks' },
    messages: { initFailed: 'Failed to initialise:', chooseDataset: 'Pick a preset dataset or upload a custom one first.', buildingIndex: 'Building index…', indexBuilt: 'Index built.', llmSaved: 'LLM config saved.', buildingEngine: 'Building query engine…', engineReady: 'Query engine ready.', chooseMetric: 'Select at least one evaluation metric.', loaded: 'Loaded {name}.', chooseJson: 'Choose a JSON file first.', uploaded: 'Uploaded {name}.', generated: 'Generated {count} QA pairs.', sampleFailed: 'Sample failed' },
  },
  zh: {
    nav: { workflow: '使用流程', features: '特性', demo: '演示', github: 'GitHub' },
    footer: { copyright: '© 2024 XRAG。保留所有权利。', github: 'GitHub', docs: '文档' },
    hero: { title: 'XRAG：eXamining the Core - Benchmarking Foundational Components in Advanced Retrieval-Augmented Generation', subtitle: '用于评估 RAG 系统基础组件的基准测试框架', demo: '在线演示' },
    steps: ['构建数据集', '构建向量库', '配置 LLM', '配置检索', '设置评测指标', '查看测试结果'],
    common: { previous: '上一步', next: '下一步', runEvaluation: '运行评测', cancel: '取消', startOver: '重新开始', loading: '加载中…' },
    dataset: {
      title: '构建数据集', preset: '选择预设数据集', custom: '或上传自定义数据集',
      descriptions: { hotpot_qa: '多跳问答数据集', drop: '数值推理数据集', natural_questions: '自然语言问答数据集' },
      folderTitle: '导入文档文件夹', folderDesc: '支持 PDF、TXT、DOCX 等格式', folderPath: '文件夹路径', folderPlaceholder: '文档文件夹路径', questionsPerFile: '每个文件生成的问题数', sentenceLength: '句子长度（-1 表示按文件处理）', outputPath: '输出 JSON 路径', chooseFolder: '从文件夹生成',
      folderFeatures: ['支持标准数据集', '支持自定义数据集集成', '支持文件夹批量导入'],
      jsonTitle: '上传 JSON 数据集', jsonDesc: '支持自定义 JSON 格式的数据集', dropJson: '拖放或选择 JSON 文件', chooseJson: '上传 JSON 文件', jsonFeatures: ['支持问答对数据集', '支持自定义元数据', '支持批量导入'],
      loaded: '已加载', documents: '篇文档', questions: '个测试问题', source: '来源', sources: { preset: '预设数据集', json: 'JSON 上传', folder: '文档文件夹' },
    },
    vector: { title: '构建向量数据库', embedding: '选择嵌入模型', splitType: '切分方式', chunkSize: '文本块大小', chunkOverlap: '文本块重叠大小', windowSize: '句子窗口大小', chunkSizes: '分层文本块大小', persistDir: '持久化目录', splitTypes: { sentence: '按句子', sentence_window: '句子窗口', character: '按字符', hierarchical: '分层切分' } },
    llm: { title: '配置大语言模型', select: '选择语言模型', apiKey: 'API 密钥', apiBase: 'API 基础地址', modelName: '模型名称', hfModel: 'HuggingFace 模型', authToken: '访问令牌', ollamaModel: 'Ollama 模型', timeout: '请求超时（秒）', temperature: '温度' },
    retrieval: { title: '配置检索', type: '检索器类型', retrieval: '检索', mode: '检索器模式', modeValue: '模式', topK: '返回结果数（Top K）', pre: '检索前处理', noneProcess: '不处理', rerank: '检索后重排', noneRerank: '不重排', orchestrator: '编排器', synthesizer: '回答合成器', qaTemplate: '问答模板', refineTemplate: '优化模板', transforms: { hyde_zeroshot: 'HyDE 零样本', hyde_fewshot: 'HyDE 少样本', stepback_zeroshot: '后退提示零样本', stepback_fewshot: '后退提示少样本' }, rerankers: { long_context_reorder: '长上下文重排序', colbertv2_rerank: 'ColBERTv2 重排', 'bge-reranker-base': 'BGE 基础重排器' } },
    metrics: { title: '设置评测指标', samples: '测试样本数', experiment: '以实验 1 模式运行（使用 cfg.test_init_total_number_documents，而非 cfg.n）', groups: { 'NLG Evaluation': '自然语言生成评测', 'LLaMA Evaluation': 'LLaMA 评测', 'DeepEval Evaluation': 'DeepEval 评测', 'UpTrain Evaluation': 'UpTrain 评测', 'SePer Evaluation': 'SePer 评测' } },
    results: { title: '查看测试结果', experimentId: '实验 ID：', running: '正在运行评测', complete: '个样本已完成。', failed: '评测失败：', cancelled: '评测已取消。', finished: '评测完成', processed: '个样本已处理。', aggregate: '汇总指标', metric: '指标', score: '得分', valid: '有效数量', perSample: '逐样本结果', answer: '回答：', expected: '预期答案：', retrieved: '已检索', chunks: '个上下文片段' },
    messages: { initFailed: '初始化失败：', chooseDataset: '请先选择预设数据集或上传自定义数据集。', buildingIndex: '正在构建索引…', indexBuilt: '索引构建完成。', llmSaved: '大语言模型配置已保存。', buildingEngine: '正在构建查询引擎…', engineReady: '查询引擎已就绪。', chooseMetric: '请至少选择一个评测指标。', loaded: '已加载 {name}。', chooseJson: '请先选择 JSON 文件。', uploaded: '已上传 {name}。', generated: '已生成 {count} 个问答对。', sampleFailed: '样本评测失败' },
  },
};
import { computed, ref, watch } from 'vue';

export function useLocale() {
  const saved = localStorage.getItem('xrag-lang');
  const initial = saved === 'en' || saved === 'zh'
    ? saved
    : (navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en');
  const lang = ref(initial);
  const t = computed(() => messages[lang.value]);

  watch(lang, value => {
    localStorage.setItem('xrag-lang', value);
    document.documentElement.lang = value === 'zh' ? 'zh-CN' : 'en';
    document.title = 'XRAG · WebUI';
  }, { immediate: true });

  const format = (text, values = {}) => Object.entries(values)
    .reduce((result, [key, value]) => result.replace(`{${key}}`, value), text);
  return { lang, t, format };
}
