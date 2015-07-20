import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.optimization.L1Updater


/**
 * Created by Bellamkonda on 6/20/2015.
 */
object SvmSgd_classification_l1 {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "Assignment1")
    val data = sc.textFile("C:/Users/Bellamkonda/Desktop/Spark/wine1.csv")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }

    // Passing parameters for L1 regularization
    val svmAlg = new SVMWithSGD()
    svmAlg.optimizer.
      setNumIterations(200).
      setRegParam(0.01).
      setUpdater(new L1Updater).setStepSize(0.1)


    // split in to training and test datasets

    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 12L)
    val training = splits(0)
    val test = splits(1)

    // Run training algorithm to build the model
    val model1 = svmAlg.run(training)

    // Evaluate model on training examples and compute training error
    val labelAndPreds = test.map { point =>
      val prediction = model1.predict(point.features)
      (point.label, prediction)
    }

    model1.clearThreshold()
    val ScoreAndLabels = test.map { case LabeledPoint(label, features) =>
      val Score = model1.predict(features)
      (Score, label)
    }
    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(ScoreAndLabels)
    val auROC = metrics.areaUnderROC()


    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / parsedData.count

    println("\nTest Error = " + testErr)
    println("\n Area under curve =" + auROC)
    println("\n Confusion Matrix=" + confusion)
  }
}