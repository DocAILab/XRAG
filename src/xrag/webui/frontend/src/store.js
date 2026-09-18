//存放全局变量
import { reactive } from 'vue';

export const store = reactive({
  step: 1, options: null, capabilities: null, config: null, error: '', notice: '', toastError: '', loading: false,
  dataset: null, presetDataset: '', presetLoading: false, uploadFile: null,
  folderPath: './data/documents', folderNum: 3, folderSentenceLen: -1,
  folderOutput: './data/generated_qa.json',
  embeddings: '', splitType: '', chunkSize: 512, chunkOverlap: 20, windowSize: 3,
  chunkSizes: '2048, 512, 128', persistDir: 'storage',
  llm: '', apiKey: '', apiBase: '', apiName: '', authToken: '', hfModel: '',
  ollamaModel: '', ollamaTimeout: 60, temperature: 0,
  orchestrator: 'default', retriever: 'BM25', retrieverMode: 0,
  queryTransform: 'none', postprocessRerank: 'none', responseSynthesizer: 'refine',
  similarityTopK: 3, textQaTemplate: '', refineTemplate: '',
  selectedMetrics: [], metricPreset: '', numSamples: 10, experiment1: false,
  experimentId: null, evalTaskId: null, evalStatus: 'idle', evalProgress: 0, evalCompleted: 0,
  evalTotal: 0, evalSamples: [], evalSummary: null, evalError: '', evalStream: null,
});
