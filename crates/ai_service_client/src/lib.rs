
tonic::include_proto!("orama_ai_service");


impl OramaModel {
    pub fn dimensions(&self) -> usize {
        match self {
            OramaModel::BgeSmall => 384,
            OramaModel::BgeBase => 768,
            OramaModel::BgeLarge => 1024,
            OramaModel::MultilingualE5Small => 384,
            OramaModel::MultilingualE5Base => 768,
            OramaModel::MultilingualE5Large => 1024,
            OramaModel::JinaEmbeddingsV2BaseCode => 768,
            OramaModel::MultilingualMiniLml12v2 => 768,
        }
    }

    pub fn senquence_length(&self) -> usize {
        //
        // From Michele slack message:
        // https://oramasearch.slack.com/archives/D0571JYV5LK/p1742488393750479
        // ```
        // intfloat/multilingual-e5-small: 512
        // intfloat/multilingual-e5-base: 512
        // intfloat/multilingual-e5-large: 512
        // BAAI/bge-small-en: 512
        // BAAI/bge-base-en: 512
        // BAAI/bge-large-en: 512
        // sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2: 128
        // jinaai/jina-embeddings-v2-base-code: 8000 (ma fai comunque massimo 512 o 1024)
        // ```
        match self {
            OramaModel::MultilingualE5Small => 512,
            OramaModel::MultilingualE5Base => 512,
            OramaModel::MultilingualE5Large => 512,
            OramaModel::BgeSmall => 512,
            OramaModel::BgeBase => 512,
            OramaModel::BgeLarge => 512,
            OramaModel::MultilingualMiniLml12v2 => 128,
            OramaModel::JinaEmbeddingsV2BaseCode => 512,
        }
    }

    pub fn overlap(&self) -> usize {
        // https://oramasearch.slack.com/archives/D0571JYV5LK/p1742488431564979
        self.senquence_length() * 2 / 100
    }
}
