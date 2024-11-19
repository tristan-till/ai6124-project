### GENERAL ###
SEED = 42

### PLOT ###
PLOT_PATH = 'runs'
PLOT_BACKBONE = 'backbone.png'
PLOT_BASELINE = 'baseline.png'
EVO_PLOT = 'evo-test.png'
FT_EVO_PLOT = 'ft-evo-test.png'
FT_PLOT_BACKBONE = 'ft-backbone.png'
FT_PLOT_BASELINE = 'ft-baseline.png'
AGG_PLOT = "agg.png"
FT_AGG_PLOT = "ft-agg.png"
CR_PLOTS = "cr"
SR_PLOTS = "sr"
MD_PLOTS = "md"
CR_EVO_PLOT = "cr-evo.png"
SR_EVO_PLOT = "sr-evo.png"
MD_EVO_PLOT = "md-evo.png"
FT_CR_EVO_PLOT = "cr-ft-evo.png"
FT_SR_EVO_PLOT = "sr-ft-evo.png"
FT_MD_EVO_PLOT = "md-ft-evo.png"
CR_INPUT_ANALYSIS = "cr-input-analysis.png"
SR_INPUT_ANALYSIS = "sr-input-analysis.png"
MD_INPUT_ANALYSIS = "md-input-analysis.png"
FT_CR_INPUT_ANALYSIS = "ft-cr-input-analysis.png"
FT_SR_INPUT_ANALYSIS = "ft-sr-input-analysis.png"
FT_MD_INPUT_ANALYSIS = "ft-md-input-analysis.png"

### DATA ###
TARGET = 'XOM'
# SUPP = ['XOM', 'OXY', 'BP', '^SPX', 'COP', 'CL=F']
SUPP = ['XOM', '^SPX']
FT_TARGET = 'CVX'
FT_SUPP = ['CVX', '^SPX']
PHASE = 60
START_DATE = '2020-01-01'
END_DATE = '2024-11-01'

### BACKBONE ###
GRU_SIZE = 256
GRU_LAYERS = 4
LSTM_SIZE = 128
LSTM_LAYERS = 4
ATTENTION_SIZE = 100
DROPOUT = 0.2
GRADIENT_CLIP = 1.0

### TRAIN ###
NUM_FOLDS = 5
NUM_EPOCHS = 250
BATCH_SIZE = 32
VAL_SIZE = 0.15
TEST_SIZE = 0.15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
SHUFFLE=True

### FINETUNE ###
FT_NUM_EPOCHS = 50

### MODEL_PATHS ###
BEST_MODEL_PATH = 'best_model.pth'
LAST_MODEL_PATH = 'last_model.pth'
FT_BEST_MODEL_PATH = 'ft_best.pth'
FT_LAST_MODEL_PATH = 'ft_last.pth'
BL_BEST_MODEL_PATH = 'bl_best.pth'
BL_LAST_MODEL_PATH = 'bl_last.pth'
FT_BL_BEST_MODEL_PATH = 'ft_bl_best.pth'
FT_BL_LAST_MODEL_PATH = 'ft_bl_last.pth'

BEST_GENOME_PATH = 'best_genome.h5'
LAST_GENOME_PATH = 'last_genome.h5'
BEST_CR_GENOME_PATH = 'best_cr_genome.h5'
LAST_CR_GENOME_PATH = 'last_cr_genome.h5'
BEST_SR_GENOME_PATH = 'best_sr_genome.h5'
LAST_SR_GENOME_PATH = 'last_sr_genome.h5'
BEST_MD_GENOME_PATH = 'best_md_genome.h5'
LAST_MD_GENOME_PATH = 'last_md_genome.h5'

### CENTROIDS ###
NUM_POINTS = 100
POW = 3

### FIS ###
NUM_IN_MF = 3
NUM_OUT_MF = 3
NUM_OUTPUTS = 3
NUM_RULES = 20
NUM_GENES = 4

### MANAGER ###
NUM_SELF_INPUTS = 1
SIGNIFICANCE_THRESHOLD = 0.15

### EVOLUTION ###
POPULATION_SIZE = 100
NUM_GENERATIONS = 100
EPISODE_LENGTH = 50
MUTATION_RATE = 0.2
ELITISM_THRESHOLD = 0.2

### FINANCIALS ###
INITIAL_CASH = 1000
INITIAL_STOCKS = 10
TRANSACTION_FEE = 0.02

### BASELINE ###
BASELINE_HIDDEN_SIZE = 64
BASELINE_HIDDEN_LAYERS = 16

### AGGREGATION ###
NUM_OBJECTIVES = 3