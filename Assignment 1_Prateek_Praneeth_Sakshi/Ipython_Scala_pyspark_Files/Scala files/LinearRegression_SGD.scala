/**
 * Created by Bellamkonda on 6/21/2015.
 */
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vectors, Vector}


object LinearRegression_SGD {
  def main(args: Array[String]) {
    val sc = new SparkContext("local[2]", "Assignment1")
    val data = sc.textFile("C:/Users/Bellamkonda/Desktop/Spark/wine_regression.csv")

    val parsedData = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(" ").map(_.toDouble)))
    }
    // split in to training and test datasets

    val splits = parsedData.randomSplit(Array(0.7, 0.3), seed = 13L)
    val training = splits(0)
    val test = splits(1)

    // Building the model
    val numIterations = 25
    val model = LinearRegressionWithSGD.train(training, numIterations)

    // Evaluate model on training examples and compute training error
    val valuesAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2)}.reduce(_ + _)/valuesAndPreds.count
    println("training Mean Squared Error = " + MSE)
   //val metrics1 = new RegressionMetrics(ScoreAndLabels)
   //val Confustion = metrics1.

  }
}
