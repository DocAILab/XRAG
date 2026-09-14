async function request(url, options, fallback) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || fallback);
  return data;
}

const jsonPost = body => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});

export const api = {
  capabilities: () => request('/api/capabilities', undefined, 'Failed to load capabilities'),
  options: () => request('/api/options', undefined, 'Failed to load options'),
  config: () => request('/api/config', undefined, 'Failed to load config'),
  updateLLM: body => request('/api/config/llm', jsonPost(body), 'LLM update failed'),
  updateVector: body => request('/api/config/vector', jsonPost(body), 'Vector update failed'),
  updateRetrieval: body => request('/api/config/retrieval', jsonPost(body), 'Retrieval update failed'),
  presetDataset: name => request('/api/dataset/preset', jsonPost({ name }), 'Failed to load dataset'),
  uploadJson(file) {
    const body = new FormData();
    body.append('file', file);
    return request('/api/dataset/upload-json', { method: 'POST', body }, 'Upload failed');
  },
  fromFolder: body => request('/api/dataset/from-folder', jsonPost(body), 'Folder generation failed'),
  buildIndex: () => request('/api/index/build', { method: 'POST' }, 'Index build failed'),
  buildQueryEngine: () => request('/api/query-engine/build', { method: 'POST' }, 'Query engine build failed'),
  startEvaluation: body => request('/api/evaluate/start', jsonPost(body), 'Failed to start evaluation'),
  cancelEvaluation: taskId => request(`/api/evaluate/${taskId}/cancel`, { method: 'POST' }, 'Failed to cancel evaluation'),
};
