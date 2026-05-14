from typing import Dict, Any

from PIL import Image
from img2vec_pytorch import Img2Vec
from chromadb import EmbeddingFunction, Embeddings
from chromadb.api.types import Images
from chromadb.utils.embedding_functions import register_embedding_function


DB_PATH = "./db"
DATA_PATH = "./data"


@register_embedding_function
class ImageEmbeddings(EmbeddingFunction):

    def __init__(self):
        self.model = Img2Vec()

    def __call__(self, input: Images) -> Embeddings:
        embeddings = self._get_imgs_embeddings(input)
        return embeddings

    def _get_imgs_embeddings(self, input):
        return [self.model.get_vec(Image.fromarray(img)) for img in input]

    @staticmethod
    def name() -> str:
        return "img2vec"

    def get_config(self) -> Dict[str, Any]:
        return dict(model=self.model)

    @staticmethod
    def build_from_config(config: Dict[str, Any]) -> "EmbeddingFunction":
        return ImageEmbeddings(config['model'])
