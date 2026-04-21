using AzureAI_IssueClassification;
using Microsoft.ML;

public class Program
{
    public static string sampledata = @"C:\Users\555397\OneDrive - Cognizant\Documents\AzureAIProject\Proj\AzureAI_IssueClassification\AzureAI_IssueClassification\SampleData\corefx_issues.tsv";
    public static void Main(string[] args)
    {
        int ch = 1;
        do
        {
            MLContext mLContext = new MLContext();
            //load data
            IDataView dataView = mLContext.Data.LoadFromTextFile<AzureAI_Issue>(path: sampledata, hasHeader: true, separatorChar: '\t', allowSparse:false);
            var trainTestSplit= mLContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            
            IDataView trainingData = trainTestSplit.TrainSet;
            IDataView testData = trainTestSplit.TestSet;
            ITransformer trainedModel = BuildAndTrainModel(mLContext, trainingData);
            EvaluateModel(mLContext, trainedModel, testData);            
            
            var aiIssue = new AzureAI_Issue()
            {
                ID = "25",
                Title = "websockets communication is very slow in  my machine",
                Description = "the websockets used under the covers by SignalR looks like going slow in my development machine",

            };
            var resultprediction = Predict(mLContext, trainedModel, aiIssue);
            Console.WriteLine($"\n==========Single Prediction==========");
            Console.WriteLine($"Title: {aiIssue.Title}\n Description:{aiIssue.Description}\n\n"+$" Prediction: {resultprediction.Area}");
            Console.WriteLine("\nDo you want to predict another text? (1 for Yes / 0 for No)");
            ch = Convert.ToInt32(Console.ReadLine());
        }
        while (ch == 1);
    }
    private static ITransformer BuildAndTrainModel(MLContext mLContext, IDataView trainingData)
    {
        //data process configuration with pipeline data transformations 
        var dataProcessPipeline = mLContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(AzureAI_Issue.Area))
            .Append(mLContext.Transforms.Text.FeaturizeText(outputColumnName: "TitleFeatures", inputColumnName: nameof(AzureAI_Issue.Title)))
            .Append(mLContext.Transforms.Text.FeaturizeText(outputColumnName: "DescriptionFeatures", inputColumnName: nameof(AzureAI_Issue.Description)))
            .Append(mLContext.Transforms.Concatenate(outputColumnName: "Features", "TitleFeatures","DescriptionFeatures"))
            .AppendCacheCheckpoint(mLContext);

        //set the training algorithm, then create and train the model
        var trainer = mLContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
        var trainingPipeline = dataProcessPipeline
            .Append(trainer)
            .Append(mLContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel")); // <-- Add this line

        //train the model fitting to the dataset
        ITransformer trainedModel = trainingPipeline.Fit(trainingData);
        return trainedModel;
    }

    //predict the result for the test dataset and evaluate the model's accuracy with MulticlassClassificationMetrics
    private static AzureAI_IssuePrediction Predict(MLContext mLContext, ITransformer trainedModel, AzureAI_Issue aiIssue)
    {
        //create prediction engine related to the loaded trained model
        var predEngine = mLContext.Model.CreatePredictionEngine<AzureAI_Issue, AzureAI_IssuePrediction>(trainedModel);
        //predict a single new sample azure ai issue
        var resultprediction = predEngine.Predict(aiIssue);
        return resultprediction;
    }

    //evaluate the model's accuracy with MulticlassClassificationMetrics
    private static void EvaluateModel(MLContext mLContext, ITransformer trainedModel, IDataView testData)
    {        
        var predictions = trainedModel.Transform(testData);
        var metrics = mLContext.MulticlassClassification.Evaluate(predictions);        
        Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:F2}");
        Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:F2}");
        Console.WriteLine($"LogLoss: {metrics.LogLoss:F2}");
    }
}