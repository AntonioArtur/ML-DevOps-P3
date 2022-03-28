# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a simple RandomForestClassifier(), the categorical columns were transformed using OnehotEncodings

## Intended Use
This particular model is used to assess my understanding of course material at udacity MLDevOps course

## Training Data
Training data is 80% of publicly available Census Bureau data

## Evaluation Data
Training data is 20% of publicly available Census Bureau data

## Metrics
precision: 0.7372754491017964
recall: 0.6265903307888041
fbeta: 0.6774415405777167

## Ethical Considerations
Is strongly advised to do a indepth search for racial biases in case this model will be applied in some real application

## Caveats and Recommendations
In case pytest does not work directly, try: python -m pytest tests/