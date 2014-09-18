package cc.factorie.app.nlp.embeddings

import scala.collection.mutable
import cc.factorie.optimize.Example
import cc.factorie.util.DoubleAccumulator
import cc.factorie.la.{DenseTensor1, WeightsMapAccumulator}


class ParagraphVectorModel (override val opts: EmbeddingOpts) extends WordEmbeddingModel(opts) {
  val sample = opts.sample.value.toDouble
  val windowLength = 10
  val rng = new util.Random()
  override def process(doc: String): Int = {

    val fullsen = doc.stripLineEnd.split(' ')

    val docId = fullsen(0).toInt
    println(docId)
    var sen = fullsen.drop(1).map(word => vocab.getId(word)).filter(id => id != -1)
    val wordCount = sen.size

    if (sample > 0)
      sen = sen.filter(id => subSample(id) != -1)
    //sen.foreach(s=>print(vocab.getWord(s)+" "))

    val senLength = sen.size
    if(senLength >= windowLength){
     for (senPosition <- 0 until (senLength / windowLength).toInt) {

       val b = rng.nextInt(senLength)
       val wordContexts = new mutable.ArrayBuffer[Int]
       val end = windowLength + b

       // make the contexts
       if(end < senLength){
        for (a <- b until end-1) {
          wordContexts += sen(a)
          //adding paragraph id

        }
        val currWord = sen(end-1)
        //println(vocab.getWord(currWord))
        val parContext = docId
        //println(wordContexts)
        //wordContexts.foreach(c => print(vocab.getWord(c)+" "))
        //println(parContext)
        trainer.processExample(new ParagraphVectorHierarchicalSoftMaxExample(this, currWord, wordContexts,parContext,vocab.getCodelength(currWord),vocab.getCode(currWord),vocab.getPoint(currWord)))
         //trainer.processExample(new ParagraphVectorNegativeSampling(this, currWord, wordContexts,parContext, 1))
         //(0 until negative).foreach(neg => trainer.processExample(new ParagraphVectorNegativeSampling(this, currWord, List(vocab.getRandWordId), -1)))
       }

     }


   }
   return wordCount
}
  def subSample(word: Int): Int = {
    val prob = vocab.getSubSampleProb(word) // pre-computed to avoid sqrt call every time.
    val alpha = rng.nextInt(0xFFFF) / 0xFFFF.toDouble
    if (prob < alpha) { return -1 }
    else return word
  }
}

/*class ParagraphVectorNegativeSampling(model: WordEmbeddingModel, word: Int, wordContexts: Seq[Int],parContext: Int) extends Example {
  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {

    val wordEmbedding = model.weights(word).value
    val contextEmbedding = new DenseTensor1(model.D, 0)
    wordContexts.foreach(context => contextEmbedding.+=(model.weights(context).value))
    contextEmbedding.+=(model.parWeights(parContext).value)
    val score: Double = wordEmbedding.dot(contextEmbedding)
    val exp: Double = math.exp(-score)
    var objective: Double = 0.0
    var factor: Double = 0.0
    if (code(d) == 1) {
      objective = -math.log1p(exp)
      factor = exp / (1 + exp)
    }
    if (code(d) == 0) {
      objective = -score - math.log1p(exp)
      factor = -1 / (1 + exp)
    }


  }
}  */



class ParagraphVectorHierarchicalSoftMaxExample(model: WordEmbeddingModel, word: Int, wordContexts: Seq[Int],parContext: Int, codeLen:Int,code: Array[Int],point:Array[Int]) extends Example {


  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator): Unit = {


    val contextEmbedding = new DenseTensor1(model.D, 0)
    wordContexts.foreach(context => contextEmbedding.+=(model.weights(context).value))
    contextEmbedding.+=(model.parWeights(parContext).value)



    for (d <- 0 until codeLen) {
      val currNode = point(d)
      //println(currNode)
      val nodeEmbedding = model.nodeWeights(currNode).value
      val score: Double = nodeEmbedding.dot(contextEmbedding)
      val exp: Double = math.exp(-score)

      var objective: Double = 0.0
      var factor: Double = 0.0
      if (code(d) == 1) {
        objective = -math.log1p(exp)
        factor = exp / (1 + exp)
      }
      if (code(d) == 0) {
        objective = -score - math.log1p(exp)
        factor = -1 / (1 + exp)
      }
      if (value ne null) value.accumulate(objective)
      if (gradient ne null) {
        wordContexts.foreach(context => gradient.accumulate(model.weights(context), nodeEmbedding, factor))
        gradient.accumulate(model.parWeights(parContext), nodeEmbedding, factor)
        gradient.accumulate(model.nodeWeights(currNode), contextEmbedding, factor)
      }

    }
  }
}