package cc.factorie.app.topics.lda
import cc.factorie._
import scala.collection.mutable._
import java.io.{PrintWriter, FileWriter, File, BufferedReader, InputStreamReader, FileInputStream}
import cc.factorie.directed._
import cc.factorie.variable._
import scala.Seq
import scala.Iterable
import scala.collection.mutable
import cc.factorie.util.SingletonIndexedSeq
import cc.factorie.maths._


// Name here must match superDoc exactly, in order to enable stitching back together again at the end
class RecursiveDocument(superDoc:Doc, val superTopic:Int) extends Document(superDoc.ws.domain, superDoc.name, Nil)
{



  var prevInTopic = false
  time = superDoc.time


  def getSuperTopicLocations(isSuperTopic:Boolean, SuperTopicLocations :ArrayBuffer[Int]) : ArrayBuffer[Int] = {
    val superdocLocations = ArrayBuffer.empty[Int]
    for (i <- 0 until superDoc.ws.length)
      if (superDoc.zs.intValue(i) == superTopic) {
        if(isSuperTopic){
          superdocLocations+=i
        }
        else{
          superdocLocations+= SuperTopicLocations(i)
        }
        ws.appendInt(superDoc.ws.intValue(i))
        if (superDoc.breaks.contains(i) || !prevInTopic) this.breaks += (ws.length-1) // preserve phrase boundary breaks, which go at the end of each word that begins a phrase
        prevInTopic = true
      } else prevInTopic = false
    ws.trimCapacity
    return superdocLocations

  }


}



class FinalDocumentList(){
  var documentList=new ArrayBuffer[RecursiveDocument]
  var termLocationSuperDoc : Buffer[ArrayBuffer[Int]]=Buffer()

  def addDocuments(doc:RecursiveDocument){
    documentList+=doc
  }

  def addLocationSuperDoc(locationArray: ArrayBuffer[Int]){
    termLocationSuperDoc+=locationArray
  }


  def getDocuments(): ArrayBuffer[RecursiveDocument]={
    return  documentList
  }

  def getLocationSuperDoc():Buffer[ArrayBuffer[Int]]={
    return termLocationSuperDoc
  }

  def getLocationSuperDocbyIndex(index:Int):ArrayBuffer[Int]={
    return termLocationSuperDoc(index)
  }
}



// Example command-lines:
// Multi-threaded, but all on one machine:
// --read-docs mytextdatadir --num-iterations=30 --fit-alpha-interval=10  --diagnostic-phrases --print-topics-phrases --write-docs recursive-lda-docs.txt
// Serialized per recursive division
// --read-docs mytextdatadir --num-iterations=30 --num-layers 1 --fit-alpha-interval=10  --diagnostic-phrases --print-topics-phrases --write-docs recursive-lda-docs.txt
// --read-docs recursive-lda-docs.txt --read-docs-topic-index 5 --num-layers 1 --num-iterations=30 --fit-alpha-interval=10  --diagnostic-phrases --print-topics-phrases --write-docs recursive-lda-docs5.txt


class RecursiveLDA(wordSeqDomain: CategoricalSeqDomain[String], numTopics: Int = 10, alpha1:Double = 0.1, beta1:Double = 0.01)(implicit model:MutableDirectedModel, override implicit val random: scala.util.Random) extends LDA(wordSeqDomain, numTopics, alpha1, beta1)(model, random)


object RecursiveLDA {
  import scala.util.control.Breaks._
  val minDocLength = 5

  def main(args:Array[String]): Unit = {
    var verbose = false
    object opts extends cc.factorie.util.DefaultCmdOptions {
      //val numTopics =     new CmdOption("num-topics", 't', 10, "N", "Number of topics at each of the two recursive levels; total number of topics will be N*N.")
      val numSuperTopics =     new CmdOption("num-topics-super", 't', 10, "N", "Number of topics at the first level")
      val numSubTopics =     new CmdOption("num-topics-sub", 't', 10, "N", "Number of topics at the second level")
      val numLayers =     new CmdOption("num-layers", 'l', 2, "N", "Number of layers of recursion in topic tree; currently only values accepted are 1 and 2.")
      val alpha =         new CmdOption("alpha", 0.1, "N", "Dirichlet parameter for per-document topic proportions.")
      val beta =          new CmdOption("beta", 0.01, "N", "Dirichlet parameter for per-topic word proportions.")
      val numThreads =    new CmdOption("num-threads", 2, "N", "Number of threads for multithreaded topic inference.")
      //val numIterations = new CmdOption("num-iterations", 'i', 50, "N", "Number of iterations of inference.")
      val diagnostic =    new CmdOption("diagnostic-interval", 'd', 100, "N", "Number of iterations between each diagnostic printing of intermediate results.")
      val diagnosticPhrases= new CmdOption("diagnostic-phrases", false, "true|false", "If true diagnostic printing will include multi-word phrases.")
      val fitAlpha =      new CmdOption("fit-alpha-interval", Int.MaxValue, "N", "Number of iterations between each re-estimation of prior on per-document topic distribution.")
      val tokenRegex =    new CmdOption("token-regex", "\\p{Alpha}+", "REGEX", "Regular expression for segmenting tokens.")
      val readDirs =      new CmdOption("read-dirs", List(""), "DIR...", "Space-(or comma)-separated list of directories containing plain text input files.")
      val readNIPS=       new CmdOption("read-nips", List(""), "DIR...", "Read data from McCallum's local directory of NIPS papers.")
      val readLines =     new CmdOption("read-lines", "", "FILENAME", "File containing lines of text, one for each document.")
      val readLinesRegex= new CmdOption("read-lines-regex", "", "REGEX", "Regular expression with parens around the portion of the line that should be read as the text of the document.")
      val readLinesRegexGroups= new CmdOption("read-lines-regex-groups", List(1), "GROUPNUMS", "The --read-lines-regex group numbers from which to grab the text of the document.")
      val readLinesRegexPrint = new CmdOption("read-lines-regex-print", false, "BOOL", "Print the --read-lines-regex match that will become the text of the document.")
      val readDocs =      new CmdOption("read-docs", "lda-docs.txt", "FILENAME", "Add documents from filename , reading document names, words and z assignments")
      val readDocsTopicIndex = new CmdOption("read-docs-topic-index", 0, "N", "Only include in this model words that were assigned to the given topic index.  (Used for disk-based parallelism.)")
      val writeDocs =     new CmdOption("write-docs", "lda-docs.txt", "FILENAME", "Save LDA state, writing document names, words and z assignments")
      val maxNumDocs =    new CmdOption("max-num-docs", Int.MaxValue, "N", "The maximum number of documents to read.")
      val printTopics =   new CmdOption("print-topics", 20, "N", "Just before exiting print top N words for each topic.")
      val printPhrases =  new CmdOption("print-topics-phrases", 20, "N", "Just before exiting print top N phrases for each topic.")
      val verboseOpt =    new CmdOption("verbose", "Turn on verbose output") { override def invoke = verbose = true }
      val numIterationsOuter =  new CmdOption("num-iterations-outer", 'i', 50, "N", "Number of iterations for the entire block sampling process.")
      val numIterationsSuper =  new CmdOption("num-iterations-super", 'i', 50, "N", "Number of iterations for inferring supertopics.")
      val numIterationsSuperFirstPass = new CmdOption("num-iterations-super-first-pass", 'i', 50, "N", "Number of iterations for inferring supertopics during first pass.")
      val numIterationsSub =  new CmdOption("num-iterations-sub", 'i', 50, "N", "Number of iterations for inferring subtopics.")
      val numIterationsBlockSampling = new CmdOption("num-iterations-block", 'i', 50, "N", "Number of iterations for inferring block moves to supertopic.")
      val superTopicOutputFile =  new CmdOption("super-topic-output", "super-topic-output.txt", "FILENAME", "File to which top words from supertopic can be saved, to be used to calculate topic coherence.")
      val subTopicOutputFile =  new CmdOption("sub-topic-output", "sub-topic-output.txt", "FILENAME", "File to which top words from subtopic can be saved, to be used to calculate topic coherence.")
      val topWordsNum = new CmdOption("num-top-words", 'i', 100, "N", "Number of top words to be printed out.")
      val testFile =   new CmdOption("test-file", "L-R-Eval.txt", "FILENAME", "Test file containing test documents to calculate L-R held-out log likelihood values.")
      // TODO Add stopwords option
    }
    val totalrunstarttime = System.currentTimeMillis
    opts.parse(args)
    implicit val random = new scala.util.Random(0)
    val numSuperTopics = opts.numSuperTopics.value
    println("No of Super Topics="+numSuperTopics)
    val numSubTopics = opts.numSubTopics.value
    println("No of Sub Topics="+numSubTopics)
    val numTopWords = opts.topWordsNum.value
    val testFile = opts.testFile.value
    object WordSeqDomain extends CategoricalSeqDomain[String]
    //implicit val model = DirectedModel()
    var lda = new RecursiveLDA(WordSeqDomain, numSuperTopics, opts.alpha.value, opts.beta.value)(DirectedModel(),random)
    val mySegmenter = new cc.factorie.app.strings.RegexSegmenter(opts.tokenRegex.value.r)
    if (opts.readDirs.wasInvoked) {
      for (directory <- opts.readDirs.value) {
        val dir = new File(directory); if (!dir.isDirectory) { System.err.println(directory+" is not a directory."); System.exit(-1) }
        println("Reading files from directory " + directory)
        breakable { for (file <- new File(directory).listFiles; if file.isFile) {
          if (lda.documents.size == opts.maxNumDocs.value) break()
          val doc = Document.fromFile(WordSeqDomain, file, "UTF-8", segmenter = mySegmenter)
          doc.time = file.lastModified
          if (doc.length >= minDocLength) lda.addDocument(doc, random)
          if (lda.documents.size % 1000 == 0) { print(" "+lda.documents.size); Console.flush() }; if (lda.documents.size % 10000 == 0) println()
        }}
        //println()
      }

      // Now that we have the full min-max range of dates, set the doc.stamps values to a 0-1 normalized value
      val dates = lda.documents.map(_.time)
      maxDate = dates.max
      minDate = dates.min
      dateRange = maxDate - minDate
      //lda.documents.foreach(doc => doc.stamps.foreach(_ := (doc.date - minDate) / dateRange))
    }
    if (opts.readNIPS.wasInvoked) {
      // A temporary hack for McCallum's development/debugging
      val directories = Range(0,13).reverse.map(i => "%02d".format(i)).take(8).map("/Users/mccallum/research/data/text/nipstxt/nips"+_)
      for (directory <- directories) {
        val year = directory.takeRight(2).toInt
        //println("RecursiveLDA directory year "+year)
        val dir = new File(directory); if (!dir.isDirectory) { System.err.println(directory+" is not a directory."); System.exit(-1) }
        println("Reading NIPS files from directory " + directory)
        for (file <- new File(directory).listFiles; if file.isFile) {
          val doc = Document.fromFile(WordSeqDomain, file, "UTF-8", segmenter = mySegmenter)
          doc.time = year
          if (doc.length >= 3) lda.addDocument(doc, random)
          print("."); Console.flush()
        }
        println()
      }
      val dates = lda.documents.map(_.time)
      maxDate = dates.max
      minDate = dates.min
      dateRange = maxDate - minDate
    }
    if (opts.readLines.wasInvoked) {
      val name = if (opts.readLines.value == "-") "stdin" else opts.readLines.value
      val source = if (opts.readLines.value == "-") scala.io.Source.stdin else scala.io.Source.fromFile(new File(opts.readLines.value))
      var count = 0
      breakable { for (line <- source.getLines()) {
        if (lda.documents.size == opts.maxNumDocs.value) break()
        val text: String =
          if (!opts.readLinesRegex.wasInvoked) line
          else {
            val textbuffer = new StringBuffer
            for (groupIndex <- opts.readLinesRegexGroups.value) {
              val mi = opts.readLinesRegex.value.r.findFirstMatchIn(line).getOrElse(throw new Error("No regex match for --read-lines-regex in "+line))
              if (mi.groupCount >= groupIndex) textbuffer append mi.group(groupIndex)
              else throw new Error("No group found with index "+groupIndex)
            }
            textbuffer.toString
          }
        if (text eq null) throw new Error("No () group for --read-lines-regex in "+line)
        if (opts.readLinesRegexPrint.value) println(text)
        val doc = Document.fromString(WordSeqDomain, name+":"+count, text, segmenter = mySegmenter)
        if (doc.length >= minDocLength) lda.addDocument(doc, random)
        count += 1
        if (count % 1000 == 0) { print(" "+count); Console.flush() }; if (count % 10000 == 0) println()
      }}
      source.close()
    }
    // On-disk representation for RecursiveLDA input/output
    if (opts.readDocs.wasInvoked) {
      val file = new File(opts.readDocs.value)
      val reader = new BufferedReader(new InputStreamReader(new FileInputStream(file)))
      reader.mark(512)
      val alphasName = reader.readLine()
      if (alphasName == "/alphas") { // If they are present, read the alpha parameters.
      val alphasString = reader.readLine(); lda.alphas.value := alphasString.split(" ").map(_.toDouble) // set lda.alphas
        reader.readLine() // consume delimiting newline
        println("Read alphas "+lda.alphas.value.mkString(" "))
      } else reader.reset() // Put the reader back to the read position when reader.mark was called
      breakable { while (true) {
        if (lda.documents.size == opts.maxNumDocs.value) break()
        val doc = new Document(WordSeqDomain, "", Nil) // doc.name will be set in doc.readNameWordsZs
        doc.zs = lda.newZs
        val filterTopicIndex = opts.readDocsTopicIndex.value
        // If readDocsTopicIndex.wasInvoked then only read in words that had been assigned readDocsTopicIndex.value, and reassign them random Zs
        val numWords = if (opts.readDocsTopicIndex.wasInvoked) doc.readNameWordsMapZs(reader, ti => if (ti == filterTopicIndex) random.nextInt(numSuperTopics) else -1) else doc.readNameWordsZs(reader)
        if (numWords < 0) break()
        else if (numWords >= minDocLength) lda.addDocument(doc, random) // Skip documents that have only one word because inference can't handle them
        else if (!opts.readDocsTopicIndex.wasInvoked) System.err.println("--read-docs skipping document %s: only %d words found.".format(doc.name, numWords))
      }}
      reader.close()
      lda.maximizePhisAndThetas
    }
    if (lda.documents.size == 0) { System.err.println("You must specific either the --input-dirs or --input-lines options to provide documents."); System.exit(-1) }
    println("\nRead "+lda.documents.size+" documents, "+WordSeqDomain.elementDomain.size+" word types, "+lda.documents.map(_.ws.length).sum+" word tokens.")

    //lda.documents.filter(_.name.endsWith("0003.txt")).foreach(d => println(d.toString)) // print example documents

    // Fit top-level LDA model
    val startTime = System.currentTimeMillis
    /*println("Documents")
    for(doc <- lda.documents){
      println(doc.ws.categoryValues)
    }*/

    val super_topic_file = new File(opts.superTopicOutputFile.value)
    val pw_super = new PrintWriter(super_topic_file)

    val sub_topic_file = new File(opts.subTopicOutputFile.value)
    //val pw_sub = new PrintWriter(sub_topic_file)
    val pw_sub = new java.io.BufferedWriter(new FileWriter(sub_topic_file))

    val super_sub_topic_file = new File("super_sub_topics.txt")
    //val pw_sub = new PrintWriter(sub_topic_file)
    val pw_super_sub = new java.io.BufferedWriter(new FileWriter(super_sub_topic_file))

    for(iterationCount <- 1 to opts.numIterationsOuter.value){
      pw_super_sub.write("Iteration "+iterationCount+"\n")
      println("Outer Loop Iteration "+iterationCount)
      println("Starting first layer inference")
      val startTimeTopLevelLDA = System.currentTimeMillis
      if(iterationCount==1) lda.inferTopics(opts.numIterationsSuperFirstPass.value, opts.fitAlpha.value, 10) else lda.inferTopics(opts.numIterationsSuper.value, opts.fitAlpha.value, 10)
      println("\nTotal Time taken for Top Level LDA for Outer Loop Iteration "+iterationCount+ " is " +  ((System.currentTimeMillis-startTimeTopLevelLDA)/1000.0)+" seconds")

      // Clear memory of parameters; we only need the documents (their words and zs)
      var documents1 = lda.documents.toSeq

      val summaries1 = Seq.tabulate(numSuperTopics)(i => lda.topicSummary(i, numTopWords))


      // [commented this part because supertopic counts required for block sampling]
      //lda = null
      //for (doc <- documents1) doc.theta = null
      //System.gc()
      val startTimeSecondLevelLDA = System.currentTimeMillis

      // Create next-level LDA models, each with its own DirectedModel, in chunks of size opts.numThreads
      //var finallda = Seq.tabulate(numSuperTopics*numSubTopics)(i => new RecursiveLDA(WordSeqDomain, numSuperTopics*numSubTopics, opts.alpha.value, opts.beta.value)(DirectedModel(),random))
      //var lda3 = Seq.tabulate(numSuperTopics*numSubTopics)(i => new RecursiveLDA(WordSeqDomain, numSubTopics*numSuperTopics, opts.alpha.value, opts.beta.value) (DirectedModel(),random))
      //var finaldocument = Seq.tabulate(numSubTopics*numSuperTopics)(i => (new ArrayBuffer[RecursiveDocument],ArrayBuffer[Int]))
      var finaldocuments = Seq.tabulate(numSubTopics*numSuperTopics)(i => new FinalDocumentList)
      var documentTopicMapFinal = new HashMap[Int,HashMap[String,ArrayBuffer[Int]]]()
      var index = 0
      var subtopicCount=0
      if (opts.numLayers.value > 1) {
        for (topicRange <- Range(0, numSuperTopics).grouped(opts.numThreads.value)) {
          var documentTopicMaplevel1 = new HashMap[Int,HashMap[String,ArrayBuffer[Int]]]()
          //var documents2 = new ArrayBuffer[RecursiveDocument]
          //object WordSeqDomain_1 extends CategoricalSeqDomain[String]

          val topicRangeStart: Int = topicRange.head
          var lda2 = Seq.tabulate(topicRange.size)(i => new RecursiveLDA(WordSeqDomain, numSubTopics, opts.alpha.value, opts.beta.value)(DirectedModel(),random))
          //for (i <- topicRange) lda2(i-topicRangeStart).diagnosticName = "Super-topic: "+summaries1(i) // So that the super-topic gets printed before each diagnostic list of subtopics

          for (doc <- documents1) {

            for (ti <- doc.zs.uniqueIntValues) if (topicRange.contains(ti)) {
              val superdocLocations = ArrayBuffer.empty[Int]
              val rdoc = new RecursiveDocument(doc, ti)
              val superDocumentLocationsPerTopic = rdoc.getSuperTopicLocations(true,superdocLocations)
              //if (rdoc.name.endsWith("0003.txt")) { println("topic="+ti+" super: "+doc.toString+"\nsub: "+rdoc.toString+"\n\n") }


              if (rdoc.ws.length > 0) {
                lda2(ti - topicRangeStart).addDocument(rdoc, random)
                //documents2 += rdoc
                //documentTopicMaplevel1.getOrElseUpdate(ti - topicRangeStart, HashMap[String,ArrayBuffer[Int]])
                //documentTopicMaplevel1.getOrElseUpdate(ti - topicRangeStart, HashMap[String,ArrayBuffer[Int]])(rdoc.name) =  superDocumentLocationsPerTopic
                if(!documentTopicMaplevel1.contains(ti - topicRangeStart)){ documentTopicMaplevel1(ti - topicRangeStart) = new HashMap[String,ArrayBuffer[Int]]()}
                documentTopicMaplevel1(ti - topicRangeStart).put(rdoc.name,superDocumentLocationsPerTopic)


              }
            }
          }

          println("Starting second-layer parallel inference for "+topicRange)
          lda2.par.foreach(_.inferTopics(opts.numIterationsSub.value, opts.fitAlpha.value, 10))

          println("Ended second-layer parallel inference for "+topicRange)

          println("Generating documents after level-2 inference")
          println("Printing the final supertopics and subtopics in iteration "+iterationCount +" for supertopic range "+topicRange.toSeq)

          //var lda3 = Seq.tabulate(numSuperTopics*numSubTopics)(i => new RecursiveLDA(WordSeqDomain, numSubTopics*numSuperTopics, opts.alpha.value, opts.beta.value) (DirectedModel(),random))

          for (i <- topicRange) {
            pw_super_sub.write("**************************************************************************************************************\n")
            val temp =  "Super-topic : "+summaries1(i)
            pw_super_sub.write(temp+"\n")
            println(temp)
            for(t <- 0 until numSubTopics){
              val subtemp = "SubTopic : "+lda2(i-topicRangeStart).topicSummary(t,numTopWords)
              pw_super_sub.write(subtemp+"\n")
              println(subtemp)
            }
            pw_super_sub.write("**************************************************************************************************************\n")
          }

          var topicid = 0
          for(lda_sub <- lda2){
            if(iterationCount == opts.numIterationsOuter.value){
              println("Printing final valid blocks")

              for(t <- 0 until numSubTopics){
                // consider only blocks which have at least one term assigned to it
                if(lda_sub.topictermcount(t)>0){
                  //val wordlist =  lda_sub.phis(t).value.top(numTopWords).map(dp => lda_sub.wordDomain.category(dp.index))
                  //scala.tools.nsc.io.File(opts.subTopicOutputFile.value).writeAll(subtopicCount.toString()+" "+lda_sub.topicWords(t, numTopWords).mkString(" ")+"\n")
                  //scala.tools.nsc.io.File(opts.subTopicOutputFile.value).appendAll(subtopicCount.toString()+" "+lda_sub.topicWords(t, numTopWords).mkString(" ")+"\n")
                  //println(subtopicCount.toString() +" "+lda_sub.topicSummary(t,numTopWords))
                  println(subtopicCount.toString()+" "+lda_sub.topicWords(t, numTopWords).mkString(" "))
                  pw_sub.write(subtopicCount.toString()+" "+lda_sub.topicWords(t, numTopWords).mkString(" ")+"\n")


                  /*for(w <- wordlist){
                    pw_sub.write(" "+ w)
                  }
                  pw_sub.write("\n")  */

                  subtopicCount+=1
                }

              }
            }
            for (doc <- lda_sub.documents){

              for (topic <- 0 to numSubTopics-1)
                for (ti <- doc.zs.uniqueIntValues){
                  if(ti==topic){

                    val rdoc = new RecursiveDocument(doc, ti)
                    val superDocumentLocations =  rdoc.getSuperTopicLocations(false,documentTopicMaplevel1(topicid)(rdoc.name))

                    if (rdoc.ws.length > 0) {
                      //finaldocument(topic+index)+=(rdoc,superDocumentLocations)

                      finaldocuments(topic+index).addDocuments(rdoc)
                      finaldocuments(topic+index).addLocationSuperDoc(superDocumentLocations)
                      //documentTopicMapFinal.getorElseUpdate(topic,HashMap[String,ArrayBuffer[Int]])(rdoc.name) = superDocumentLocations

                      //if(!(documentTopicMapFinal.contains(topic))){documentTopicMapFinal(topic) = new HashMap[String,ArrayBuffer[Int]]()}
                      //documentTopicMapFinal(topic).put(rdoc.name,superDocumentLocations)
                    }

                  }
                }
            }
            index += numSubTopics
            topicid+=1

          }






        }
        /*println("LDA final documents")
        for(doc <- finaldocuments){

          if(doc.getDocuments().size!=0){
            println(doc.getLocationSuperDoc())
          for(doc1 <- doc.getDocuments()){

              println("Doc "+doc1.ws.categoryValues)

          }
          }
        }
        println("Top level LDA")
        for(doc <- lda.documents){
          println("Doc "+doc.ws.categoryValues)
          println("Topic assignments "+doc.zs.intValues.toSeq)
        } */
        if(iterationCount == opts.numIterationsOuter.value){
          pw_sub.close()
        }
        println("\nTotal Time taken for Second Level LDA for all supertopics for Outer Loop Iteration "+iterationCount+ " is " +  ((System.currentTimeMillis-startTimeSecondLevelLDA)/1000.0)+" seconds")
        println("Starting block sampling")
        val startTime = System.currentTimeMillis
        // Call Block Sampling Code
        val numTopicChanges = BlockSampling(lda,finaldocuments,opts.numIterationsBlockSampling.value,numSuperTopics,numTopWords,subtopicCount)
        println("\nTotal Time taken for Block Sampling for Outer Loop Iteration "+iterationCount+ " is " +  ((System.currentTimeMillis-startTime)/1000.0)+" seconds")
        /*for (subLda <- lda2; ti <- 0 until numTopics) {
        val tp = timeMeanAlphaBetaForTopic(subLda.documents, ti) // time parameters
        println("%s  mean=%g variance=%g".format(subLda.topicSummary(ti), tp._1, tp._2))
        }*/
      }




      /* documents1 = null // To allow garbage collection, but note that we now loose word order in lda3 documents
       println("Finished in "+(System.currentTimeMillis - startTime)+" ms.")
       // Re-assemble the documents, and optionally write them out.
       val documents3 = new HashMap[String,Document]
       object ZDomain3 extends DiscreteDomain(numTopics * numTopics)
       object ZSeqDomain3 extends DiscreteSeqDomain { def elementDomain = ZDomain3 }
       class Zs3 extends DiscreteSeqVariable { def domain = ZSeqDomain3 }
       while (documents2.size > 0) {
         val doc = documents2.last
         val doc3 = documents3.getOrElseUpdate(doc.name, { val d = new Document(WordSeqDomain, doc.name, Nil); d.zs = new Zs3; d })
         val ws = doc.ws; val zs = doc.zs
         var i = 0; val len = doc.length
         while (i < len) {
           val zi = doc.superTopic * numTopics + zs.intValue(i)
           val wi = ws.intValue(i)
           doc3.ws.appendInt(wi)
           doc3.zs.appendInt(zi)
           if (doc.breaks.contains(i)) doc3.breaks += (doc3.ws.length-1) // preserve phrase boundaries
           i += 1
         }
         documents2.remove(documents2.size-1) // Do this to enable garbage collection as we create more doc3's
       }
       documents1 = documents3.values.toSeq // Put them back into documents1 so that they can be written out below.


     if (opts.writeDocs.wasInvoked) {
       println("\nWriting state to "+opts.writeDocs.value)
       val file = new File(opts.writeDocs.value)
       val pw = new PrintWriter(file)
       pw.println("/alphas")
       pw.println(Seq.fill(numTopics * numTopics)(opts.alpha.value).mkString(" ")) // Just set all alphas to 1.0 // TODO can we do better?
       pw.println()
       documents1.foreach(_.writeNameWordsZs(pw))
       pw.close()
     }

      */

    }



    for(i <- 0 to numSuperTopics-1){
      val wordlist =  lda.phis(i).value.top(numTopWords).map(dp => lda.wordDomain.category(dp.index))
      pw_super.write(i.toString())

      for(w <- wordlist){
        pw_super.write(" "+ w)
      }
      pw_super.write("\n")
    }

    pw_super.close()
    pw_super_sub.close()

    object ZDomain_1 extends DiscreteDomain(numSuperTopics)
    var alphas: Array[Double]=null
    alphas = Array.fill[Double](numSuperTopics)((lda.alphas.value.massTotal)/numSuperTopics)
    val alphaSum = lda.alphas.value.massTotal
    val beta = lda.beta1
    val betaSum = beta*lda.wordDomain.size
    val phiCounts = new DiscreteMixtureCounts(lda.wordDomain, lda.ZDomain)
    for (doc<- lda.documents){
      phiCounts.incrementFactor(lda.model.parentFactor(doc.ws).asInstanceOf[PlatedCategoricalMixture.Factor], 1)
    }
    var termTopicCounts = Array.ofDim[Int](lda.wordDomain.size,numSuperTopics)
    var topicCounts : Array[Int]=null
    topicCounts = Array.fill[Int](numSuperTopics)(0)
    for(topic <- 0 to numSuperTopics-1){
      topicCounts(topic) = phiCounts.mixtureCounts(topic)
    }

    var typeTopicCounts= new Array[HashMap[Int, Int]](lda.wordDomain.size)

    for (wi <- 0 until lda.wordDomain.size)  {
      var topicCounts = new HashMap[Int,Int]()

      val phiCountsWi = phiCounts(wi)
      var tp = 0
      while(tp < phiCountsWi.numPositions){
        val topic = phiCountsWi.indexAtPosition(tp)
        val count =  phiCountsWi.countAtPosition(tp)
        topicCounts(topic) = count
        termTopicCounts(wi)(topic) = count

        tp=tp+1
      }


      typeTopicCounts(wi)=  topicCounts


    }
    // Max likelihood calculation
    println("Final Loglikelihood "+modelLogLikelihood(lda,numSuperTopics,topicCounts,termTopicCounts))
    val HeldOutLLREval  = new LREval(lda.wordSeqDomain,ZDomain_1,alphas,alphaSum,beta,betaSum,topicCounts,typeTopicCounts)
    println( "HeldOutLikelihood for the test document is " + HeldOutLLREval.calcLR(testFile,5,false))
    println("Total time taken for the whole run is "+  (System.currentTimeMillis-totalrunstarttime)/(1000.0*60.0*60.0) + " hours")
  }



  // Code for time-stamp parameterization
  // Related to Topics-over-Time [Wang, McCallum, KDD 2006]

  // These globals are set above in main opts.readDirs.wasInvoked
  var maxDate: Long = 0
  var minDate: Long = 0
  var dateRange: Double = 0.0

  /** Convert from Long doc.time to stamp falling in 0...1 range */
  def time2Stamp(t:Long): Double = {
    val result = (t - minDate) / dateRange
    assert(result >= 0.0, "input=%d minDate=%d dateRange=%g result=%g".format(t, minDate, dateRange, result))
    assert(result <= 1.0, result)
    result
  }
  /** Calculate Beta distribution parameters (alpha, beta) for the topicIndex. */
  def timeMeanAlphaBetaForTopic(documents:Iterable[Doc], topicIndex:Int): (Double, Double, Double, Double) = {
    val stamps = new util.DoubleArrayBuffer
    for (d <- documents; z <- d.zs.intValues; if z == topicIndex) {
      if (d.time < 0) throw new Error(d.name+" has year "+d.time)
      stamps += time2Stamp(d.time)
    }
    val mean = maths.sampleMean(stamps)
    val variance = maths.sampleVariance(stamps, mean)
    //println("RecursiveLDA.timeMeanAlphaBeta min=%d max=%d range=%g".format(minDate, maxDate, dateRange))
    //println("RecursiveLDA.timeMeanAlphaBeta mean=%g variance=%g".format(mean, variance))
    (mean, variance, MaximizeBetaByMomentMatching.maxAlpha(mean, variance), MaximizeBetaByMomentMatching.maxBeta(mean, variance))
  }

  def BlockSampling(superLDA:LDA,subdocuments:Seq[FinalDocumentList] ,numIterations:Int,numSuperTopics:Int,numTopWords:Int,subtopicCount:Int)={

    var localTopicCounts = new Array[Int](numSuperTopics)
    var localTermTopicCounts = Array.ofDim[Int](superLDA.wordDomain.size,numSuperTopics)
    var docTopicCounts = new HashMap[String,HashMap[Int,Int]]()

    var t1 =  System.currentTimeMillis


    val phiCounts = new DiscreteMixtureCounts(superLDA.wordDomain, superLDA.ZDomain)

    var superldaiter = superLDA.documents.iterator
    while (superldaiter.hasNext){
      val doc = superldaiter.next()
      phiCounts.incrementFactor(superLDA.model.parentFactor(doc.ws).asInstanceOf[PlatedCategoricalMixture.Factor], 1)
    }
    superldaiter = superLDA.documents.iterator
    while (superldaiter.hasNext){
      val doc = superldaiter.next()
      val docName = doc.name

      if(!docTopicCounts.contains(docName)){ docTopicCounts(docName) = new HashMap[Int,Int]()}
      var zp=0
      val docLen = doc.zs.length
      while(zp < docLen){
        val zi = doc.zs.intValue(zp)
        if(!docTopicCounts(docName).contains(zi)) docTopicCounts(docName).put(zi,0)
        docTopicCounts(docName).put(zi,(docTopicCounts(docName)(zi))+1)
        zp=zp+1
      }
      var topic=0
      while(topic < numSuperTopics){
        if(!docTopicCounts(docName).contains(topic)) docTopicCounts(docName).put(topic,0)
        topic=topic+1
      }

      //val len = doc.ws.categoryValues.length
      val len = doc.ws.length


      var i=0
      while (i < len) {
        val wi = doc.ws.intValue(i)
        val phiCountsWi = phiCounts(wi)
        var tp = 0
        while(tp < phiCountsWi.numPositions){
          val topic = phiCountsWi.indexAtPosition(tp)
          localTermTopicCounts(wi)(topic) = phiCountsWi.countAtPosition(tp)

          tp=tp+1
        }
        i=i+1
      }


    }
    var topic=0
    while (topic< numSuperTopics){
      localTopicCounts(topic) = phiCounts.mixtureCounts(topic)
      topic = topic + 1
    }



    var t2 =  System.currentTimeMillis
    //println("Time for initialization step "+(t2-t1)/1000.0 +" seconds")

    val beta1 = superLDA.beta1
    val betasum = beta1*superLDA.wordDomain.size
    val alpha = (superLDA.alphas.value.massTotal)/numSuperTopics
    var i = 1
    println("Loglikelihood of model before starting block sampling = "+modelLogLikelihood(superLDA,numSuperTopics,localTopicCounts,localTermTopicCounts))
    while(i<= numIterations){


      var numTopicChanges = 0
      println("Block Sampling Iteration "+i)
      val startTime = System.currentTimeMillis
      var count=0

      val subdocIter = subdocuments.iterator
      while(subdocIter.hasNext){
        val subdoc = subdocIter.next()
        if((subdoc.getDocuments().size)!=0){


          var oldtopic = -1
          var foundoldtopic = false

          var docIndex=0

          t1 =  System.currentTimeMillis

          val iter = subdoc.getDocuments().iterator
          while(iter.hasNext){
            val doc = iter.next()
            val docName = doc.name
            val SuperDocLocation = subdoc.getLocationSuperDocbyIndex(docIndex)
            //val len = doc.ws.categoryValues.length

            val len = doc.ws.length

            assert(len==SuperDocLocation.size)
            var i=0
            while (i < len) {

              val loc = SuperDocLocation(i)
              val wi = superLDA.getDocument(docName).ws.intValue(loc)
              val zi =  superLDA.getDocument(docName).zs.intValue(loc)


              //superLDA.getDocument(doc.name).theta.value.masses.+=(zi,-1.0)
              docTopicCounts(docName)(zi) =  docTopicCounts(docName)(zi) -1
              localTermTopicCounts(wi)(zi) = localTermTopicCounts(wi)(zi) - 1
              localTopicCounts(zi) = localTopicCounts(zi)-1

              i+=1
              if(!foundoldtopic) {oldtopic = zi ; foundoldtopic=true}

            }
            docIndex+=1
          }
          t2 =  System.currentTimeMillis
          //println("Time for decrement step "+(t2-t1)/1000.0 +" seconds")
          t1 =  System.currentTimeMillis
          var topicScores = new ArrayBuffer[Double](numSuperTopics)
          var maxTopicLogScore =  Double.NegativeInfinity
          t1 =  System.currentTimeMillis
          var topic=0
          while(topic < numSuperTopics){
            var topicLogScore = 0.0
            var topicSizeIncrement = 0
            var typeIncrement = new HashMap[Int,Int]().withDefaultValue(0)
            docIndex=0
            val subdocIter = subdoc.getDocuments().iterator
            while(subdocIter.hasNext){
              val doc=  subdocIter.next()
              val docName = doc.name
              val SuperDocLocation = subdoc.getLocationSuperDocbyIndex(docIndex)
              val docTopicCount = docTopicCounts(docName)(topic)
              var docIncr = 0
              //val len = doc.ws.categoryValues.length
              val len = doc.ws.length

              var i=0
              val docNameWs = superLDA.getDocument(docName).ws
              while (i < len) {
                val loc = SuperDocLocation(i)
                val wi =  docNameWs.intValue(loc)

                topicLogScore = topicLogScore + Math.log(docTopicCount + docIncr + alpha) +
                  Math.log(localTermTopicCounts(wi)(topic) + typeIncrement(wi) + superLDA.beta1) -
                  Math.log( localTopicCounts(topic) + topicSizeIncrement + betasum)
                docIncr+=1

                typeIncrement.update(wi,typeIncrement(wi)+1)
                topicSizeIncrement=topicSizeIncrement + 1
                i+=1
              }
              docIndex+=1
            }

            topicScores+= topicLogScore
            maxTopicLogScore = Math.max(maxTopicLogScore, topicLogScore)
            topic= topic+1

          }

          var sumTerm = 0.0
          var t=0
          while(t < numSuperTopics){
            sumTerm = sumTerm + Math.exp(topicScores(t) - maxTopicLogScore)
            t = t+1
          }
          var logSumProb = maxTopicLogScore + Math.log(sumTerm)


          var sumProb = 0.0
          var tp=0
          while(tp < numSuperTopics){
            topicScores(tp) = Math.exp(topicScores(tp) - logSumProb)
            sumProb = sumProb + topicScores(tp)
            tp=tp+1
          }


          var sample = random.nextDouble() * sumProb

          var newTopic = -1
          while (sample > 0.0) {
            newTopic+=1
            sample = sample - topicScores(newTopic)
          }

          t2 =  System.currentTimeMillis
          // println("Time for sampling step "+(t2-t1)/1000.0 +" seconds")
          docIndex=0
          var block=new ArrayBuffer[String]
          t1 =  System.currentTimeMillis

          var docSizeIter = subdoc.getDocuments().iterator
          while(docSizeIter.hasNext){
            val doc = docSizeIter.next()
            val docName = doc.name
            val SuperDocLocation = subdoc.getLocationSuperDocbyIndex(docIndex)

            //val len = doc.ws.categoryValues.length
            doc.ws.categoryValues.foreach(e => block+=e)
            val len = doc.ws.length

            val docNameWs = superLDA.getDocument(docName).ws
            var i=0
            while (i < len) {
              val loc = SuperDocLocation(i)
              val wi = docNameWs.intValue(loc)

              //superLDA.getDocument(doc.name).theta.value.masses.+=(newTopic,1.0)
              docTopicCounts(docName)(newTopic) = docTopicCounts(docName)(newTopic) + 1
              localTopicCounts(newTopic) =  localTopicCounts(newTopic) + 1
              localTermTopicCounts(wi)(newTopic) = localTermTopicCounts(wi)(newTopic) + 1
              superLDA.getDocument(docName).zs.asInstanceOf[superLDA.Zs].set(loc,newTopic)(null)

              i+=1

            }
            docIndex+=1

          }
          t2 =  System.currentTimeMillis
          //println("Time for increment step "+(t2-t1)/1000.0 +" seconds")

          // have to modify the code to print the updated supertopics.
          val summaries1 = Seq.tabulate(numSuperTopics)(i => superLDA.topicSummary(i, numTopWords))
          val final_output = new ArrayBuffer[String]

          block.groupBy(e=>e).mapValues(_.length).toSeq.sortBy(_._2).takeRight(numTopWords).foreach(l => final_output prepend l._1)
          if(oldtopic!=newTopic)
          {
            numTopicChanges=numTopicChanges+1

            println("BLOCK MOVE : "+ final_output.mkString(" ") +" FROM SUPERTOPIC "+oldtopic+ "->"+newTopic)
            println("Supertopic  :"+summaries1(oldtopic))
            println("Supertopic  :"+summaries1(newTopic))

            println("Loglikelihood of model after block move = "+modelLogLikelihood(superLDA,numSuperTopics,localTopicCounts,localTermTopicCounts))

          }



          //numTopicChanges +=newtopics
          count+=1
        }

      }

      println("No of block moves for "+count+" blocks after Iteration " + i+" is "+numTopicChanges)
      println("Time taken for Block Sampling Iteration "+i +" is "+ ((System.currentTimeMillis-startTime)/1000.0)+" seconds")
      println("Loglikelihood of model after block sampling iteration "+i+" = "+modelLogLikelihood(superLDA,numSuperTopics,localTopicCounts,localTermTopicCounts))
      i=i+1
    }
    superLDA.phis.foreach(_.value.zero())
    var wi=0
    while (wi < superLDA.wordDomain.size)  {
      //(0 until numSuperTopics).foreach(ti => superLDA.phis(ti).value.masses.+=(wi, beta1))
      var ti=0
      while(ti < numSuperTopics){
        superLDA.phis(ti).value.masses.+=(wi, localTermTopicCounts(wi)(ti))
        ti=ti+1
      }
      wi = wi+1
    }

    var docIter = superLDA.documents.iterator
    while (docIter.hasNext) {
      val doc = docIter.next()
      val theta = doc.theta
      theta.value.zero()
      for (dv <- doc.zs.discreteValues) theta.value.masses.+=(dv.intValue, 1.0)

    }



  }

  def modelLogLikelihood(lda:LDA,numTopics:Int,localTopicCounts:Array[Int],localTermTopicCounts:Array[Array[Int]]):Double={

    var logLikelihood=0.0
    var topicLogGammas = new Array[Double](numTopics)
    var topicCounts = new Array[Int](numTopics)
    var alpha = lda.alphas.value.toArray
    val alphaSum = lda.alphas.value.massTotal
    val beta = lda.beta1
    val numTypes = lda.wordDomain.size
    var tCount=0
    while(tCount<numTopics){
      topicLogGammas(tCount)=logGamma(alpha(tCount))
      tCount=tCount+1
    }
    val docIter = lda.documents.iterator
    while(docIter.hasNext){

      val doc = docIter.next()
      val len = doc.ws.length
      var i=0
      while (i < len) {
        val zi = doc.zs.intValue(i)
        topicCounts(zi) = topicCounts(zi)+1
        i+=1
      }
      tCount=0
      while(tCount<numTopics){
        if(topicCounts(tCount)>0){
          logLikelihood+= logGamma(alpha(tCount) + topicCounts(tCount)) - topicLogGammas(tCount)
        }
        tCount=tCount+1
      }

      logLikelihood-= logGamma(alphaSum+len)
      topicCounts = Array.fill[Int](numTopics)(0)
    }
    logLikelihood += lda.documents.size * logGamma(alphaSum)


    var nonZeroTypeTopics=0.0
    var wi=0


    while (wi < numTypes)  {
      val localCounts = localTermTopicCounts(wi)
      var tp = 0
      while(tp < numTopics){
        val count = localCounts(tp)
        if(count>0){
          nonZeroTypeTopics+=1
          logLikelihood+=logGamma(beta+count)
        }
        tp=tp+1

      }
      wi+=1
    }

    tCount=0
    while(tCount<numTopics){
      logLikelihood-=logGamma((beta*numTypes)+localTopicCounts(tCount))
      tCount+=1
    }
    logLikelihood+=logGamma(numTypes * beta) * numTopics
    logLikelihood-=logGamma(beta) * nonZeroTypeTopics
    return logLikelihood


  }



}