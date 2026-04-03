import logging
from typing import Dict, Any

from preprocessing_agent import PreprocessingAgent
from risk_scoring_agent import RiskScoringAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PipelineService:
    def __init__(
        self,
        csv_path: str,
        target_column: str,
        test_size: float = 0.2,
        random_state: int = 42,
        smote_ratio: float = 1.0,
        model_dir: str = 'models',
    ):
        self.csv_path = csv_path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.smote_ratio = smote_ratio
        self.model_dir = model_dir

    def run_pipeline(self) -> Dict[str, Any]:
        try:
            logger.info('Starting pipeline with PreprocessingAgent')
            preprocessor = PreprocessingAgent(
                csv_path=self.csv_path,
                target_column=self.target_column,
                test_size=self.test_size,
                random_state=self.random_state,
            )

            split_data = preprocessor.process()

            logger.info('Starting RiskScoringAgent')
            risk_agent = RiskScoringAgent(
                data=split_data,
                test_size=self.test_size,
                random_state=self.random_state,
                smote_ratio=self.smote_ratio,
                model_dir=self.model_dir,
            )

            result = risk_agent.run()

            logger.info('Pipeline completed successfully')
            return {
                'status': 'success',
                'model_metrics': result.get('metrics'),
                'predictions': result.get('predictions'),
                'best_model_name': result.get('best_model', {}).get('best_model_name'),
                'error': None,
            }

        except Exception as ex:
            logger.error('Pipeline failed: %s', ex, exc_info=True)
            return {
                'model_metrics': {},
                'predictions': None,
                'status': 'failure',
                'error': str(ex),
                'best_model_name': None,
            }
