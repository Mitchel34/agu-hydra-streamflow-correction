"""Model construction and predictor-only tensor interfaces."""
from .config import CANDIDATES, ModelConfig
from .inputs import MaskedFeatures, ModelInputs
from .architectures import Seq2SeqLSTM, HydraGRUConv, TemporalTransformer, build_model
from .losses import normalized_residual_loss
