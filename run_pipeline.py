import logging

from pipeline_service import PipelineService
from pipeline_schemas import PipelineRequest, PipelineResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    request = PipelineRequest(
        csv_path='UCI_Credit_Card.csv',
        target_column='default.payment.next.month',
        test_size=0.2,
        random_state=42,
        smote_ratio=1.0,
        model_dir='models',
    )

    service = PipelineService(
        csv_path=request.csv_path,
        target_column=request.target_column,
        test_size=request.test_size,
        random_state=request.random_state,
        smote_ratio=request.smote_ratio,
        model_dir=request.model_dir,
    )

    result = service.run_pipeline()

    response = PipelineResponse(
        status=result.get('status', 'failure'),
        model_metrics=result.get('model_metrics'),
        predictions=result.get('predictions'),
        best_model_name=result.get('best_model_name'),
        error=result.get('error'),
    )

    logger.info('Pipeline response: %s', response)
    print(response)


if __name__ == '__main__':
    main()
