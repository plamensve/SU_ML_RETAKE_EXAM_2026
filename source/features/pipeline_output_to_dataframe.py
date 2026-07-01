import pandas as pd


def pipeline_output_to_dataframe(pipeline, transformed_array, original_data):
    """
    :param pipeline: SKLEARN PIPELINE
    :param transformed_array: numpy.ndarray
    :param original_data: PD.DATAFRAME
    :return: PD.DATAFRAME
    """
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

    data_encoded = pd.DataFrame(
        transformed_array,
        columns=feature_names,
        index=original_data.index
    )

    data_encoded.columns = (
        data_encoded.columns
        .str.replace('one_hot_encoder__', '', regex=False)
        .str.replace('remainder__', '', regex=False)
        .str.lower()
    )

    return data_encoded