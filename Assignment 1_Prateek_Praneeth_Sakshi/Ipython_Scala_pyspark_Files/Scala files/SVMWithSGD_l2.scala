import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.{MulticlassMetrics, BinaryClassificationMetrics}

/**
 * Created by Bellamkonda on 6/20/2015.
 */
object SvmSgd_classification_l2 {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "Assignment1")
    val data = sc.textFile("C:/Users/Bellamkonda/Desktop/Spark/wine1.csv")
    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }
    // split in to training and test datasets

    val splits = parsedData.randomSplit(Array(0.7, 0.3), seed = 12L)
    val training = splits(0)
    val test = splits(1)

    // Run training algorithm to build the model
    val numIterations = 20
    val model = SVMWithSGD.train(training, numIterations)

    // Evaluate model on training examples and compute training error
    val labelAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(labelAndPreds)
    val auROC = metrics.areaUnderROC()
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / parsedData.count

    println("\nTest Error = " + testErr)
    println("\n Area under curve =" + auROC)
  }
}

