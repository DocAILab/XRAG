export const quickMockResult = {
  experimentId: 'mock-quick-hotpotqa-001',
  evalStatus: 'done',
  evalProgress: 1,
  evalCompleted: 10,
  evalTotal: 10,
  evalMeta: {
    preset: 'quick',
    dataset: 'HotpotQA',
    duration: '02:18',
    model: 'gpt-4.1-mini',
  },
  evalSummary: {
    n: 10,
    global: {
      F1: 0.7426,
      mrr: 0.8117,
      hit10: 0.9320,
      NDCG: 0.8463,
    },
    metrics: {
      NLG_chrf_pp: { score: 0.6841, valid_count: 10 },
      NLG_rouge_rougeL: { score: 0.7128, valid_count: 10 },
      NLG_wer: { score: 0.2386, valid_count: 10 },
    },
  },
  evalSamples: [
    {
      index: 0,
      question: 'Which city is the birthplace of the author of The Hobbit?',
      actual_response: 'J. R. R. Tolkien was born in Bloemfontein, South Africa.',
      expected_answer: 'Bloemfontein',
      retrieval_context: ['J. R. R. Tolkien was an English writer born in Bloemfontein.', 'The Hobbit was written by J. R. R. Tolkien.', 'Bloemfontein is one of South Africa\'s capital cities.'],
    },
    {
      index: 1,
      question: 'What nationality is the director of The Shape of Water?',
      actual_response: 'The director, Guillermo del Toro, is Mexican.',
      expected_answer: 'Mexican',
      retrieval_context: ['The Shape of Water was directed by Guillermo del Toro.', 'Guillermo del Toro is a Mexican filmmaker.', 'The film was released in 2017.'],
    },
    {
      index: 2,
      question: 'Which river runs through the capital city of France?',
      actual_response: 'The Seine runs through Paris, the capital of France.',
      expected_answer: 'The Seine',
      retrieval_context: ['Paris is the capital and largest city of France.', 'The Seine flows through Paris.', 'The river reaches the English Channel at Le Havre.'],
    },
  ],
};
