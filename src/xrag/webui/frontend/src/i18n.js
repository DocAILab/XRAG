export const messages = {
  en: {
    nav: { workflow: 'Workflow', features: 'Features', demo: 'Demo', github: 'GitHub' },
    footer: { copyright: '© 2024 XRAG. All rights reserved.', github: 'GitHub', docs: 'Docs', ram: 'RAM: ' },
    hero: {
      title: 'XRAG: eXamining the Core - Benchmarking Foundational Components in Advanced Retrieval-Augmented Generation',
      subtitle: 'A powerful benchmarking framework for evaluating foundational components in advanced RAG systems',
      demo: 'Online Demo',
    },
    steps: ['Build Dataset', 'Build Vector Database', 'Configure LLM', 'Configure Retrieval', 'Set Evaluation Metrics', 'View Test Results'],
    common: { previous: 'Previous', next: 'Next', runEvaluation: 'Run Evaluation', cancel: 'Cancel', close: 'Close', startOver: 'Start Over', loading: 'Loading…' },
    dataset: {
      title: 'Build Dataset', preset: 'Choose Preset Dataset', custom: 'Or Upload Custom Dataset', loadSelected: 'Load Selected Dataset', loadingPreset: 'Loading dataset',
      descriptions: { hotpot_qa: 'Multi-hop QA Dataset', drop: 'Numerical Reasoning Dataset', natural_questions: 'Natural Language QA Dataset' },
      folderTitle: 'Upload Document Folder', folderDesc: 'Supports PDF, TXT, DOCX, etc.', folderPath: 'Folder path', folderPlaceholder: 'Path to a folder of documents', questionsPerFile: 'Number of questions per file', sentenceLength: 'Sentence length (-1 = file-level)', outputPath: 'Output JSON path', chooseFolder: 'Generate from Folder',
      folderFeatures: ['Standard dataset support', 'Custom dataset integration', 'Folder import support'],
      jsonTitle: 'Upload JSON Dataset', jsonDesc: 'Supports custom JSON format datasets', dropJson: 'Drop or pick a JSON file', chooseJson: 'Upload JSON File', jsonFeatures: ['Supports QA pair datasets', 'Supports custom metadata', 'Supports batch import'],
      loaded: 'Loaded', documents: 'documents', questions: 'test questions', source: 'source', sources: { preset: 'preset', json: 'JSON upload', folder: 'document folder' },
    },
    vector: { title: 'Build Vector Database', embedding: 'Embedding Model', splitType: 'Split Type', chunkSize: 'Chunk Size', chunkOverlap: 'Chunk Overlap', windowSize: 'Sentence Window Size', chunkSizes: 'Hierarchy Chunk Sizes', persistDir: 'Persist Directory', splitTypes: {} },
    llm: { title: 'Configure LLM', select: 'Select Language Model Endpoint type', apiKey: 'API Key', apiBase: 'API Base URL', modelName: 'Model Name', queryModels: 'Fetch models from URL', modelsFound: 'Found {count} models.', modelsQueryFailed: 'Automatic model lookup failed. Enter a model name manually.', hfModel: 'HuggingFace Model', authToken: 'Auth Token', ollamaModel: 'Ollama Model', timeout: 'Request Timeout (seconds)', temperature: 'Temperature' },
    retrieval: {
      title: 'Configure Retrieval', type: 'Retriever Type', retrieval: 'Retrieval', mode: 'Retriever Mode', modeValue: 'Mode', topK: 'Top K Results', pre: 'Pre-retrieval Process', noneProcess: 'No Processing', rerank: 'Post-process Rerank', noneRerank: 'No Reranking', orchestrator: 'Orchestrator', synthesizer: 'Response Synthesizer', qaTemplate: 'QA Template', refineTemplate: 'Refine Template', rerankers: { long_context_reorder: 'Long context reorder (LlamaIndex)', colbertv2_rerank: 'colbert-ir/colbertv2.0', 'bge-reranker-base': 'BAAI/bge-reranker-base' }
    },
    metrics: {
      title: 'Set Evaluation Metrics', samples: 'Number of test samples', presets: 'Recommended presets', selected: 'metrics selected', highCost: 'High cost', experimental: 'Experimental', requires: 'Requires:',
      presetNames: { quick: 'Quick', paper: 'Paper Main Experiment', complete: 'Complete', custom: 'Custom' },
      groups: { conr: 'Retrieval Quality', cong: 'Answer Text Matching', cogl: 'Semantic Evaluation', advanced: 'Advanced Evaluation' },
      sections: { retrieval: 'Retrieval Quality', text_matching: 'Text Matching', context_quality: 'Retrieval Context Quality', answer_quality: 'Answer Quality', grounding: 'Context and Answer Consistency', retrieval_utility: 'Retrieval Utility', golden_context: 'Golden Context Diagnostics' },
      metricLabels: { SePer_with_context: 'SePer with context', SePer_without_context: 'SePer without context', SePer_delta: 'SePer delta' },
      seperOutputs: 'Outputs: with context, without context, and delta.', seperUnavailable: 'Unavailable because SePer is disabled in the current configuration.', goldenContextNote: 'G uses the golden context as the reference.',
    },
    results: { title: 'View Test Results', experimentId: 'Experiment ID:', running: 'Running evaluation', complete: 'samples complete.', failed: 'Evaluation failed:', cancelled: 'Evaluation cancelled.', finished: 'Evaluation complete', processed: 'samples processed.', retrievalAggregate: 'Retrieval Quality (ConR)', aggregate: 'Other Metrics', metric: 'Metric', score: 'Score', valid: 'Valid Count', unavailable: 'Unavailable', perSample: 'Per-sample results', answer: 'Answer:', expected: 'Expected:', retrieved: 'Retrieved', chunks: 'context chunks' },
    messages: { initFailed: 'Failed to initialise:', chooseDataset: 'Pick a preset dataset or upload a custom one first.', datasetNotFound: 'Dataset {name} was not found.', datasetLoadFailed: 'Could not load dataset {name}. Please ensure it is available.', buildingIndex: 'Building index…', indexBuilt: 'Index built.', llmSaved: 'LLM config saved.', buildingEngine: 'Building query engine…', engineReady: 'Query engine ready.', chooseMetric: 'Select at least one evaluation metric.', loaded: 'Loaded {name}.', chooseJson: 'Choose a JSON file first.', uploaded: 'Uploaded {name}.', generated: 'Generated {count} QA pairs.', sampleFailed: 'Sample failed' },
  },
  zh: {
    nav: { workflow: '使用流程', features: '特性', demo: '演示', github: 'GitHub' },
    footer: { copyright: '© 2026 XRAG。保留所有权利。', github: 'GitHub', docs: '文档', ram: '内存：' },
    hero: { title: 'XRAG: eXamining the Core - Benchmarking Foundational Components in Advanced Retrieval-Augmented Generation', subtitle: '用于评估 RAG 系统基础组件的基准测试框架', demo: '在线演示' },
    steps: ['构建数据集', '构建向量库', '配置 LLM', '配置检索', '设置评测指标', '查看测试结果'],
    common: { previous: '上一步', next: '下一步', runEvaluation: '运行评测', cancel: '取消', close: '关闭', startOver: '重新开始', loading: '加载中…' },
    dataset: {
      title: '构建数据集', preset: '选择预设数据集', custom: '或上传自定义数据集', loadSelected: '加载所选数据集', loadingPreset: '正在加载数据集',
      descriptions: { hotpot_qa: '多跳问答数据集', drop: '数值推理数据集', natural_questions: '自然语言问答数据集' },
      folderTitle: '导入文档文件夹', folderDesc: '支持 PDF、TXT、DOCX 等格式', folderPath: '文件夹路径', folderPlaceholder: '文档文件夹路径', questionsPerFile: '每个文件生成的问题数', sentenceLength: '句子长度（-1 表示按文件处理）', outputPath: '输出 JSON 路径', chooseFolder: '从文件夹生成',
      folderFeatures: ['支持标准数据集', '支持自定义数据集集成', '支持文件夹批量导入'],
      jsonTitle: '上传 JSON 数据集', jsonDesc: '支持自定义 JSON 格式的数据集', dropJson: '拖放或选择 JSON 文件', chooseJson: '上传 JSON 文件', jsonFeatures: ['支持问答对数据集', '支持自定义元数据', '支持批量导入'],
      loaded: '已加载', documents: '篇文档', questions: '个测试问题', source: '来源', sources: { preset: '预设数据集', json: 'JSON 上传', folder: '文档文件夹' },
    },
    vector: { title: '构建向量数据库', embedding: '嵌入模型', splitType: '切分方式', chunkSize: '文本块大小', chunkOverlap: '文本块重叠大小', windowSize: '句子窗口大小', chunkSizes: '分层文本块大小', persistDir: '持久化目录', splitTypes: { sentence: '按句子', sentence_window: '句子窗口', character: '按字符', hierarchical: '分层切分' } },
    llm: { title: '配置大语言模型', select: '选择LLM接入端点类型', apiKey: 'API 密钥', apiBase: 'API Base URL', modelName: '模型名称', queryModels: '从URL读取模型列表', modelsFound: '已找到 {count} 个模型。', modelsQueryFailed: '自动查询失败，请手动填写模型名称。', hfModel: 'HuggingFace 模型', authToken: '访问令牌', ollamaModel: 'Ollama 模型', timeout: '请求超时（秒）', temperature: '温度' },
    retrieval: {
      title: '配置检索', type: '检索器类型', retrieval: '检索', mode: '检索器模式', modeValue: '模式', topK: '返回结果数（Top K）', pre: '检索前处理', noneProcess: '无', rerank: '检索后重排', noneRerank: '无', orchestrator: '编排器', synthesizer: '回答合成器', qaTemplate: '问答模板', refineTemplate: '优化模板', rerankers: { long_context_reorder: 'Long context reorder (LlamaIndex)', colbertv2_rerank: 'colbert-ir/colbertv2.0', 'bge-reranker-base': 'BAAI/bge-reranker-base' }
    },
    metrics: {
      title: '设置评测指标', samples: '测试样本数', presets: '推荐预设', selected: '个指标已选择', highCost: '高成本', experimental: '实验性', requires: '需要：',
      presetNames: { quick: '快速评测', paper: '论文主实验', complete: '完整评测', custom: '自定义' },
      groups: { conr: '检索质量', cong: '答案文本匹配', cogl: '语义评测', advanced: '高级评测' },
      sections: { retrieval: '检索质量', text_matching: '文本匹配', context_quality: '检索上下文质量', answer_quality: '回答质量', grounding: '上下文与回答一致性', retrieval_utility: '检索效用', golden_context: '黄金上下文诊断' },
      metricLabels: { SePer_with_context: 'SePer（有上下文）', SePer_without_context: 'SePer（无上下文）', SePer_delta: 'SePer 差值' },
      seperOutputs: '输出：有上下文、无上下文和差值。', seperUnavailable: '当前配置未启用 SePer，因此不可选择。', goldenContextNote: 'G 表示使用黄金上下文作为参考。',
    },
    results: { title: '查看测试结果', experimentId: '实验 ID：', running: '正在运行评测', complete: '个样本已完成。', failed: '评测失败：', cancelled: '评测已取消。', finished: '评测完成', processed: '个样本已处理。', retrievalAggregate: '检索质量（ConR）', aggregate: '其他评测指标', metric: '指标', score: '得分', valid: '有效数量', unavailable: '不可用', perSample: '逐样本结果', answer: '回答：', expected: '预期答案：', retrieved: '已检索', chunks: '个上下文片段' },
    messages: { initFailed: '初始化失败：', chooseDataset: '请先选择预设数据集或上传自定义数据集。', datasetNotFound: '未找到{name}数据集。', datasetLoadFailed: '{name}数据集加载失败，请确保数据集存在。', buildingIndex: '正在构建索引…', indexBuilt: '索引构建完成。', llmSaved: '大语言模型配置已保存。', buildingEngine: '正在构建查询引擎…', engineReady: '查询引擎已就绪。', chooseMetric: '请至少选择一个评测指标。', loaded: '已加载 {name}。', chooseJson: '请先选择 JSON 文件。', uploaded: '已上传 {name}。', generated: '已生成 {count} 个问答对。', sampleFailed: '样本评测失败' },
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
