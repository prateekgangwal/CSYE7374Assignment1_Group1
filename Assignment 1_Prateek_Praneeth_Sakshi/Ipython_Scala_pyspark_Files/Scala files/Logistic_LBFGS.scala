/**
 * Created by Bellamkonda on 6/21/2015.
 */
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

object Logistic_LBFGS {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "Assignment1")
    val data = sc.textFile("C:/Users/Bellamkonda/Desktop/Spark/wine1.csv")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }
    // split in to training and test datasets

    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val training = splits(0)
    val test = splits(1)
    // Run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS()
      .setNumClasses(10)
      .run(training)

    // Compute raw scores on the test set.
    val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
      val prediction = model.predict(features)
      (prediction, label)
    }

    model.clearThreshold()
    val ScoreAndLabels = test.map { case LabeledPoint(label, features) =>
      val Score = model.predict(features)
      (Score, label)
    }
    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(ScoreAndLabels)
    val auROC = metrics.areaUnderROC()

    val testErr = predictionAndLabels.filter(r => r._1 != r._2).count.toDouble / parsedData.count
    println("\nTest Error = " + testErr)
    println("\n Area under curve =" + auROC)

  }
}



