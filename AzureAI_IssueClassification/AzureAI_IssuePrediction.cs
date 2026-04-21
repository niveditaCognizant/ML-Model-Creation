using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace AzureAI_IssueClassification
{
    public class AzureAI_IssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area { get; set; }
    }
}
