/* Copyright (C) 2008-2014 University of Massachusetts Amherst.
   This file is part of "FACTORIE" (Factor graphs, Imperative, Extensible)
   http://factorie.cs.umass.edu, http://github.com/factorie
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
package cc.factorie.app.nlp.coref.mention

import scala.collection.mutable.HashSet
import scala.collection.mutable.ArrayBuffer
import cc.factorie.app.nlp._
import cc.factorie.app.nlp.coref._
import cc.factorie.app.nlp.phrase.{Phrase,NounPhraseType}
import cc.factorie.app.nlp.parse.ParseTree
import cc.factorie.app.nlp.ner._
import cc.factorie.app.nlp.pos.PennPosTag
import scala.Some


class ParseBasedMentionList(spans:Iterable[Mention]) extends MentionList(spans)
//class NerSpanList extends TokenSpanList[NerSpan]

@deprecated("This functionality should be moved to cc.factorie.app.nlp.coref.ParseBasedMentionPhraseFinder.")
object ParseBasedMentionFinding extends ParseBasedMentionFinding(false)
@deprecated("This functionality should be moved to cc.factorie.app.nlp.coref.ParseBasedMentionPhraseFinder.")
object ParseAndNerBasedMentionFinding extends ParseBasedMentionFinding(true)

@deprecated("This functionality should be moved to cc.factorie.app.nlp.coref.ParseBasedMentionPhraseFinder.")
class ParseBasedMentionFinding(val useNER: Boolean) extends DocumentAnnotator {

  private final val PERSONAL_PRONOUNS = Seq("PRP", "PRP$")
  private final val COMMON_NOUNS      = Seq("NN" , "NNS")
  private final val PROPER_NOUNS      = Seq("NNP", "NNPS")
  private final val ALL_NOUNS         = Seq("NN","NNS","NNP","NNPS","PRP","PRP$")

  private def isPersonalPronoun(t: Token) = PERSONAL_PRONOUNS.contains(t.posTag.categoryValue.toUpperCase)
  private def isCommonNoun     (t: Token) = COMMON_NOUNS.contains(t.posTag.categoryValue.toUpperCase)
  private def isProperNoun     (t: Token) = PROPER_NOUNS.contains(t.posTag.categoryValue.toUpperCase)
  private def isNoun           (t: Token) = ALL_NOUNS.contains(t.posTag.categoryValue.toUpperCase)

  def predictMentionType(t: Token): Option[String] =
    if(isPersonalPronoun(t)) Some("PRO")
    else if(isCommonNoun(t)) Some("NOM")
    else if(isProperNoun(t)) Some("NAM")
    else None

  var FILTER_APPOS = true /* This flag is so that appositive filtering can be turned off.
                            If the mentions that we are extracting do not include the appositives as part of a mention
                            we want to make sure that we are extracting the appositives separately
                            default behavior is that we do filter the appositives.   */


  private def nerSpans(doc: Document): Seq[Mention] = {
    val coref = doc.getCoref
    (for (span <- doc.attr[ConllNerSpanBuffer]) yield
      coref.addMention(new Phrase(span.section, span.start, span.length, span.length - 1)) //this sets the head token idx to be the last token in the span
      ).toSeq
  }

  private def NNPSpans(doc : Document) : Seq[Mention] = {
    val coref = doc.getCoref
    val spans = ArrayBuffer[ArrayBuffer[Token]]()
    spans += ArrayBuffer[Token]()
    for(section <- doc.sections; sentence <- section.sentences; token <- sentence.tokens) {
      if(spans.last.nonEmpty && spans.last.last.next != token) spans += ArrayBuffer[Token]()
      if(isProperNoun(token)) spans.last += token
    }
    if(spans.nonEmpty && spans.last.isEmpty) spans.remove(spans.length-1)
    (for(span <- spans) yield
      coref.addMention(new Phrase(span.head.section, span.head.positionInSection, span.last.positionInSection-span.head.positionInSection+1, span.last.positionInSection-span.head.positionInSection))).toSeq
  }

  // [Assumes personal pronouns are single tokens.]
  private def personalPronounSpans(doc: Document): Seq[Mention] = {
    val coref = doc.getCoref
    (for (section <- doc.sections; s <- section.sentences;
           (t,i) <- s.tokens.zipWithIndex if isPersonalPronoun(t)) yield
        coref.addMention(new Phrase(section, s.start + i, 1,0))
      ).toSeq
  }

  private def getHeadTokenIdx(m: Mention): Int = {
   val tokenIdxInSection =  getHead(
      m.phrase.head.sentence.parse,
      m.phrase.start until (m.phrase.start + m.phrase.length) //these are section-level offsets
    )
    val tokenIdxInSpan = tokenIdxInSection - m.phrase.start
    assert(tokenIdxInSpan >= 0 && tokenIdxInSpan <= m.phrase.length)
    tokenIdxInSpan
  }
  //this expects as input indices in the **document section** not the sentence
  //note that this never returns the root as the head, it always returns a pointer to an actual token in the sentence
  //it will either return the root of a parse tree span, or a token that is a child of the root
  def getHead(parse: ParseTree, subtree: Seq[Int]): Int = {
    val sentenceLevelIndices = subtree.map(i => i - parse.sentence.start)
    var curr = sentenceLevelIndices.head
    val leftBoundary = sentenceLevelIndices.head
    val rightBoundary = sentenceLevelIndices.last
    while(parse.parentIndex(curr) > 0 && containedInInterval(leftBoundary,rightBoundary,parse.parentIndex(curr))){
      curr = parse.parentIndex(curr)
    }
    curr + parse.sentence.start  //this shifts it back to have section-level indices
  }

  private def containedInInterval(left: Int, right: Int, testIndex: Int): Boolean = {
    testIndex >= left && testIndex <= right
  }

  final val copularVerbs = collection.immutable.HashSet[String]() ++ Seq("is","are","was","'m")

  final val allowedChildLabels = Set("amod", "det", "nn", "num", "hmod", "hyph", "possessive", "poss", "predet", "nmod", "dep")
  final val disallowedChildLabels = Set("conj", "punct", "prep", "cc", "appos", "npadvmod", "advmod", "quantmod", "partmod", "rcmod", "dobj", "nsubj", "infmod", "ccomp", "advcl", "aux", "intj", "neg", "preconj", "prt", "meta", "parataxis", "complm", "mark")

  private def nounPhraseSpans(doc: Document, nounFilter: Token => Boolean): Seq[Mention] =  {
    val mentions = ArrayBuffer[Mention]()
    val coref = doc.getCoref
    for (section <- doc.sections; s <- section.sentences; (t, si) <- s.tokens.zipWithIndex if nounFilter(t);
         label = s.parse.label(t.positionInSentence).categoryValue
         if label != "nn" && label != "hmod")  {
      val children = s.parse.children(t.positionInSentence)
      children.foreach(c => {
        val cat = s.parse.label(c.positionInSentence).categoryValue
        if (!(allowedChildLabels.contains(cat) || disallowedChildLabels.contains(cat))) {
          println("BAD LABEL: " + cat)
          // println(doc.owplString(DepParser1))
        }
      })
      val goodChildren = children.filter(c => allowedChildLabels.contains(s.parse.label(c.positionInSentence).categoryValue))
      val tokens = Seq(t) ++ goodChildren.map(c => s.parse.subtree(c.positionInSentence)).flatten
      val sTokens = tokens.sortBy(_.positionInSection)
      val start = sTokens.head.positionInSection
      val end = sTokens.last.positionInSection
      mentions += coref.addMention(new Phrase(section, start, end-start+1, sTokens.zipWithIndex.filter(i => i._1 eq t).head._2))
    }
    mentions
  }


  private def dedup(mentions: Seq[Mention]): Seq[Mention] = {
      def dedupOverlappingMentions(mentions: Seq[Mention]): Mention = {
        if(mentions.length == 1){
          return mentions.head
        }else{
          mentions.find(_.phrase.attr[NounPhraseType].categoryValue == "NAM").getOrElse(mentions.head)
        }
      }


      mentions
      .groupBy(m => (m.phrase.section, m.phrase.start, m.phrase.length))
      .values.map(mentionSet => dedupOverlappingMentions(mentionSet)).toSeq
      .sortBy(m => (m.phrase.tokens.head.stringStart, m.phrase.length))

  }


  def process(doc: Document): Document = {
    // The mentions are all added to coref in the methods
    personalPronounSpans(doc)
    nounPhraseSpans(doc, isCommonNoun)
    nounPhraseSpans(doc, isProperNoun)
    NNPSpans(doc)
    doc
  }

  def prereqAttrs: Iterable[Class[_]] = if (!useNER) Seq(classOf[parse.ParseTree]) else Seq(classOf[parse.ParseTree], classOf[ner.IobConllNerTag])
  def postAttrs: Iterable[Class[_]] = Seq(classOf[MentionList])
  override def tokenAnnotationString(token:Token): String = token.document.attr[MentionList].filter(mention => mention.phrase.contains(token)) match { case ms:Seq[Mention] if ms.length > 0 => ms.map(m => m.phrase.attr[NounPhraseType].categoryValue+":"+m.phrase.indexOf(token)).mkString(","); case _ => "_" }


}


