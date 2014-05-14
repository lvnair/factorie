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

import cc.factorie.app.nlp.coref._
import cc.factorie.app.nlp.phrase._
import cc.factorie.app.nlp.wordnet.WordNet
import cc.factorie.app.nlp.{Token, TokenSpan}
import cc.factorie.app.strings.Stopwords
import scala.collection.mutable
import cc.factorie.app.nlp.phrase.{Number, Gender}

//object CorefMention{
//  def mentionToCorefMention(m: Mention): CorefMention = {
//    val cm = new CorefMention(m, m.phrase.start, m.phrase.sentence.indexInSection)
//    //cm.attr += new MentionEntityType(m, m.attr[MentionEntityType].categoryValue)
//    cm
//  }
//
//
//  //todo: the default headTokenIndex here is a little funky. If you don't pass it anything, it sets it using the NN heuristic below.
//  //todo: however, if the span truly has the root as its parse parent, we also use the NN heuristic. Is there something else we should be doing for this second case?
//  def apply(span: TokenSpan, tokenNum: Int, sentenceNum: Int, mentionType: String = null, headTokenIndex: Int = -1) = {
//    //here, we use the final noun as the head if it wasn't passed a headTokenIndex (from parsing)
//    val headInd = {
//      if(headTokenIndex > span.length){
//        throw new IllegalStateException("the constructor expects headTokenIndex to be an offset from the beginning of the span, not the document")
//      }
//      if(headTokenIndex == -1){
//        val idx = span.value.lastIndexWhere(_.posTag.categoryValue.startsWith("NN"))
//        //assert(idx != -1, "failed to find a noun in the span") //todo: put this back in
//        if(idx < 0)  span.length -1 else idx      //todo: this handles the case where it didn't find an NN anywhere. Should that be happening?
//      }else
//        headTokenIndex
//    }
//    val docMention = span.document.getCoref.addMention(new Phrase(span, headInd))
//    docMention.phrase.attr += new NounPhraseType(docMention.phrase, mentionType)
//    new CorefMention(docMention, tokenNum,  sentenceNum)
//  }
//
//  val posTagsSet = Set("PRP", "PRP$", "WP", "WP$")
//
//  val properSet = Set("NNP", "NNPS")
//
//  val nounSet = Seq("NN", "NNS")
//
//  val posSet = Seq("POS")
//}



//// TODO Most of these linguistic feature methods should get moved to Phrase.
//// If coref needs caching, they should be cached somewhere in coref code. -akm
//class CorefMention(val mention: Mention, val tokenNum: Int, val sentenceNum: Int) extends cc.factorie.util.Attr {
//  val _head =  mention.phrase.tokens(mention.phrase.headTokenOffset)  //here, the head token index is an offset into the span, not the document
//  def headToken: Token = _head
//  def parentEntity = mention.entity // TODO Get rid of this method, and just use "entity" method -akm
//  def mType = headToken.posTag.categoryValue
//  def span = mention.phrase
//  //def entityType: String = mention.attr[MentionEntityType].categoryValue
//  def entityType: String = mention.phrase.attr[OntonotesEntityType].categoryValue // TODO Should return just EntityType, not String. -akm
//  def document = mention.phrase.document
//
//  val isPRO = CorefMention.posTagsSet.contains(mType)
//  val isProper = CorefMention.properSet.contains(mention.phrase.headToken.posTag.categoryValue)
//  val isNoun = CorefMention.nounSet.contains(mention.phrase.headToken.posTag.categoryValue)
//  val isPossessive = CorefMention.posSet.contains(mention.phrase.headToken.posTag.categoryValue)
//
//  def isAppositionOf(m : CorefMention) : Boolean = {
//    val isAppo = headToken.parseLabel.categoryValue == "appos"
//    val isChildOf = headToken.parseParent == m.headToken
//    isAppo && isChildOf
//  }
//
//  var cache = new MentionCharacteristics(this)
//  def clearCache() {
//    cache = new MentionCharacteristics(this)
//  }
//
//  def hasSpeakWord: Boolean = cache.hasSpeakWord
//  def gender = cache.gender
//  def number = cache.number
//  def nonDeterminerWords: Seq[String] = cache.nonDeterminerWords
//  def acronym: Set[String] = cache.acronym
//  def nounWords: Set[String] = cache.nounWords
//  def lowerCaseHead: String = cache.lowerCaseHead
//  def initials: String = cache.initials
//  def predictEntityType: String = cache.predictEntityType
//  def headPhraseTrim: String = cache.headPhraseTrim
//  def demonym: String = cache.demonym
//  def capitalization: Char = cache.capitalization
//  def wnLemma: String = cache.wnLemma
//  def wnSynsets = cache.wnSynsets
//  def wnHypernyms = cache.wnHypernyms
//  def wnAntonyms = cache.wnAntonyms
//
//  def printInfo : String = Seq[String]("gender", gender,"number", number,"nondet",nonDeterminerWords.mkString(" "),"acronym",acronym.mkString(" "),"nounwords",nounWords.mkString(" "),"lowercasehead",lowerCaseHead,"initials",initials,"ent-type",predictEntityType,"head-phase-trim",headPhraseTrim,"capitalization",capitalization.toString,"wnlemma",wnLemma).mkString("\n")
//}

/** Various lazily-evaluated cached characteristics of a Mention, typically attached to a Mention as an attr. */
class MentionCharacteristics(val mention: Mention) {
  import cc.factorie.app.nlp.lexicon
  // TODO These should be cleaned up and made more efficient -akm
  lazy val isPRO = CorefFeatures.posTagsSet.contains(mention.phrase.headToken.posTag.categoryValue)
  lazy val isProper = CorefFeatures.properSet.contains(mention.phrase.headToken.posTag.categoryValue)
  lazy val isNoun = CorefFeatures.nounSet.contains(mention.phrase.headToken.posTag.categoryValue)
  lazy val isPossessive = CorefFeatures.posSet.contains(mention.phrase.headToken.posTag.categoryValue)

//  def isAppositionOf(m:Mention) : Boolean = {
//    val isAppo = mention.phrase.headToken.parseLabel.categoryValue == "appos"
//    val isChildOf = mention.phrase.headToken.parseParent == m.phrase.headToken
//    isAppo && isChildOf
//  }

  lazy val hasSpeakWord = mention.phrase.exists(s => lexicon.iesl.Say.contains(s.string))
  lazy val wnLemma = WordNet.lemma(mention.phrase.headToken.string, "n")
  lazy val wnSynsets = WordNet.synsets(wnLemma).toSet
  lazy val wnHypernyms = WordNet.hypernyms(wnLemma)
  lazy val wnAntonyms = wnSynsets.flatMap(_.antonyms()).toSet
  lazy val nounWords: Set[String] =
      mention.phrase.tokens.filter(_.posTag.categoryValue.startsWith("N")).map(t => t.string.toLowerCase).toSet
  lazy val lowerCaseHead: String = mention.phrase.headToken.string.toLowerCase // was: mention.phrase.toLowerCase, but this looks buggy to me. -akm
  lazy val headPhraseTrim: String = mention.phrase.tokensString(" ").trim
  lazy val nonDeterminerWords: Seq[String] =
    mention.phrase.tokens.filterNot(_.posTag.categoryValue == "DT").map(t => t.string.toLowerCase)
  lazy val initials: String =
      mention.phrase.tokens.map(_.string).filterNot(lexicon.iesl.OrgSuffix.contains).filter(t => t(0).isUpper).map(_(0)).mkString("")
  //lazy val predictEntityType: String = m.mention.attr[MentionEntityType].categoryValue
  lazy val predictEntityType: String = mention.phrase.attr[OntonotesEntityType].categoryValue // TODO Why not just name this "entityTypeCategory"? And we should use the intValue instead anyway! -akm?
  lazy val demonym: String = lexicon.iesl.DemonymMap.getOrElse(headPhraseTrim, "")

  lazy val capitalization: Char = {
      if (mention.phrase.length == 1 && mention.phrase.head.positionInSentence == 0) 'u' // mention is the first word in sentence
      else { // This was missing before, and I think this was a serious bug. -akm
        val s = mention.phrase.value.filter(_.posTag.categoryValue.startsWith("N")).map(_.string.trim) // TODO Fix this slow String operation
        if (s.forall(_.forall(_.isUpper))) 'a'
        else if (s.forall(t => t.head.isLetter && t.head.isUpper)) 't'
        else 'f'
      }
    }
  lazy val gender = mention.phrase.attr[Gender].categoryValue
  lazy val number = mention.phrase.attr[Number].categoryValue
  lazy val nounPhraseType = mention.phrase.attr[NounPhraseType].categoryValue
  lazy val genderIndex = mention.phrase.attr[Gender].intValue // .toString // TODO Why work in terms of String instead of Int? -akm
  lazy val numberIndex = mention.phrase.attr[Number].intValue // .toString
  lazy val nounPhraseTypeIndex = mention.phrase.attr[NounPhraseType].intValue
  lazy val headPos = mention.phrase.headToken.posTag.categoryValue

  lazy val acronym: Set[String] = {
    if (mention.phrase.length == 1)
        Set.empty
      else {
        val alt1 = mention.phrase.value.map(_.string.trim).filter(_.exists(_.isLetter)) // tokens that have at least one letter character
        val alt2 = alt1.filterNot(t => Stopwords.contains(t.toLowerCase)) // alt1 tokens excluding stop words
        val alt3 = alt1.filter(_.head.isUpper) // alt1 tokens that are capitalized
        val alt4 = alt2.filter(_.head.isUpper)
        Seq(alt1, alt2, alt3, alt4).map(_.map(_.head).mkString.toLowerCase).toSet
      }
  }
}

// TODO I think this should be renamed, but I'm not sure to what. -akm
object CorefFeatures {
  val posTagsSet = Set("PRP", "PRP$", "WP", "WP$")
  val properSet = Set("NNP", "NNPS")
  val nounSet = Seq("NN", "NNS")
  val posSet = Seq("POS")
  
  def getPairRelations(s1: Mention, s2: Mention): String = {
    val l1 = s1.phrase.headToken.string.toLowerCase
    val l2 = s2.phrase.headToken.string.toLowerCase
    val s1c = s1.attr[MentionCharacteristics]
    val s2c = s2.attr[MentionCharacteristics]
    if (l1 == l2)
      "match"
    else if (l1.contains(l2) || l2.contains(l1))
      "substring"
    else if (s1c.wnSynsets.exists(a => s2c.wnSynsets.contains(a)))
      "Syn"
    else if (s1c.wnSynsets.exists(a => s2c.wnHypernyms.contains(a)) || s2c.wnSynsets.exists(a => s1c.wnHypernyms.contains(a)))
      "Hyp"
    else if (s1c.wnSynsets.exists(s2c.wnAntonyms.contains))
      "Ant"
    else
      "Mismatch"
  }

  def matchingTokensRelations(m1:Mention, m2:Mention) = {
    import cc.factorie.app.nlp.lexicon
    val set = new mutable.HashSet[String]()
    val m1c = m1.attr[MentionCharacteristics]
    val m2c = m2.attr[MentionCharacteristics]
    for (w1 <- m2.phrase.toSeq.map(_.string.toLowerCase))
      for (w2 <- m1.phrase.toSeq.map(_.string.toLowerCase))
       if (w1.equals(w2) || m2c.wnSynsets.exists(m1c.wnHypernyms.contains) || m1c.wnHypernyms.exists(m2c.wnHypernyms.contains) ||
           lexicon.iesl.Country.contains(w1) && lexicon.iesl.Country.contains(w2) ||
           lexicon.iesl.City.contains(w1) && lexicon.iesl.City.contains(w2) ||
           lexicon.uscensus.PersonFirstMale.contains(w1) && lexicon.uscensus.PersonFirstMale.contains(w2) ||
           // commented out the femaleFirstNames part, Roth publication did not use
           lexicon.uscensus.PersonFirstFemale.contains(w1) && lexicon.uscensus.PersonFirstFemale.contains(w2) ||
           lexicon.uscensus.PersonLast.contains(w1) && lexicon.uscensus.PersonLast.contains(w2))
        set += getPairRelations(m1, m2)
    set.toSet
  }

  def countCompatibleMentionsBetween(m1:Mention, m2:Mention, mentions:Seq[Mention]): Seq[String] = {
    val doc = m1.phrase.document
    val ments = mentions.filter(s => s.phrase.start < m1.phrase.start && s.phrase.start > m2.phrase.end)
    val iter = ments.iterator
    var numMatches = 0
    while (numMatches <= 2 && iter.hasNext) {
      val m = iter.next()
      if (CorefFeatures.gendersMatch(m, m1).equals("t") && CorefFeatures.numbersMatch(m, m1).equals("t")) numMatches += 1
    }

    if (numMatches <= 2) (0 to numMatches).map(_.toString)
    else (0 to numMatches).map(_.toString) :+ "_OVER2"
  }

  val maleHonors = Set("mr", "mister")
  val femaleHonors = Set("ms", "mrs", "miss", "misses")
  val neuterWN = Set("artifact", "location", "group")

  val malePron = Set("he", "him", "his", "himself")
  val femalePron = Set("she", "her", "hers", "herself")
  val neuterPron = Set("it", "its", "itself", "this", "that", "anything", "something",  "everything", "nothing", "which", "what", "whatever", "whichever")
  val personPron = Set("you", "your", "yours", "i", "me", "my", "mine", "we", "our", "ours", "us", "myself", "ourselves", "themselves", "themself", "ourself", "oneself", "who", "whom", "whose", "whoever", "whomever", "anyone", "anybody", "someone", "somebody", "everyone", "everybody", "nobody")

  val allPronouns = maleHonors ++ femaleHonors ++ neuterWN ++ malePron ++ femalePron ++ neuterPron ++ personPron
  // TODO: this cache is not thread safe if we start making GenderMatch not local
  // val cache = scala.collection.mutable.Map[String, Char]()
  import cc.factorie.app.nlp.lexicon
  def namGender(m: Mention): Char = {
    val fullhead = m.phrase.string.trim.toLowerCase // TODO Is this change with "string" correct? -akm 2/28/2014
    var g = 'u'
    val words = fullhead.split("\\s")
    if (words.length == 0) return g

    val word0 = words.head
    val lastWord = words.last

    var firstName = ""
    var honor = ""
    if (lexicon.iesl.PersonHonorific.contains(word0)) {
      honor = word0
      honor = removePunct(honor)
      if (words.length >= 3)
        firstName = words(1)
    } else if (words.length >= 2) {
      firstName = word0
    } else {
      firstName = word0
    }

    // determine gender using honorifics
    if (maleHonors.contains(honor))
      return 'm'
    else if (femaleHonors.contains(honor))
      return 'f'

    // determine from first name
    if (lexicon.uscensus.PersonFirstMale.contains(firstName))
      g = 'm'
    else if (lexicon.uscensus.PersonFirstFemale.contains(firstName))
      g = 'f'
    else if (lexicon.uscensus.PersonLast.contains(lastWord))
      g = 'p'

    if (lexicon.iesl.City.contains(fullhead) || lexicon.iesl.Country.contains(fullhead)) {
      if (g.equals("m") || g.equals("f") || g.equals("p"))
        return 'u'
      g = 'n'
    }

    if (lexicon.iesl.OrgSuffix.contains(lastWord)) {
      if (g.equals("m") || g.equals("f") || g.equals("p"))
        return 'u'
      g = 'n'
    }

    g
  }

  def nomGender(m: Mention, wn: WordNet): Char = {
    val fullhead = m.phrase.string.toLowerCase
    if (wn.isHypernymOf("male", fullhead))
      'm'
    else if (wn.isHypernymOf("female", fullhead))
      'f'
    else if (wn.isHypernymOf("person", fullhead))
      'p'
    else if (neuterWN.exists(wn.isHypernymOf(_, fullhead)))
      'n'
    else
      'u'
  }


  def proGender(m: Mention): Char = {
    val pronoun = m.phrase.string.toLowerCase
    if (malePron.contains(pronoun))
      'm'
    else if (femalePron.contains(pronoun))
      'f'
    else if (neuterPron.contains(pronoun))
      'n'
    else if (personPron.contains(pronoun))
      'p'
    else
      'u'
  }


  def strongerOf(g1: Char, g2: Char): Char = {
    if ((g1 == 'm' || g1 == 'f') && (g2 == 'p' || g2 == 'u'))
      g1
    else if ((g2 == 'm' || g2 == 'f') && (g1 == 'p' || g1 == 'u'))
      g2
    else if ((g1 == 'n' || g1 == 'p') && g2 == 'u')
      g1
    else if ((g2 == 'n' || g2 == 'p') && g1 == 'u')
      g2
    else
      g2
  }

  // TODO Do we really want to return a Char here?
  def gendersMatch(m1:Mention, m2:Mention): Char = {
    val g1 = m2.phrase.attr[Gender].intValue
    val g2 = m1.phrase.attr[Gender].intValue
    import GenderDomain._
    // TODO This condition could be much simplified 
    if (g1 == UNKNOWN || g2 == UNKNOWN)
      'u'
    else if (g1 == PERSON && (g2 == MALE || g2 == FEMALE || g2 == PERSON))
      'u'
    else if (g2 == PERSON && (g1 == MALE || g1 == FEMALE || g1 == PERSON))
      'u'
    else if (g1 == g2)
      't'
    else
      'f'
  }

  def headWordsCross(m1:Mention, m2:Mention, model: PairwiseCorefModel): String = {
    val w1 = m2.attr[MentionCharacteristics].headPhraseTrim
    val w2 = m1.attr[MentionCharacteristics].headPhraseTrim
    val rare1 = 1.0 / model.MentionPairLabelThing.tokFreq.getOrElse(w1.toLowerCase, 1).toFloat > 0.1
    val rare2 = 1.0 / model.MentionPairLabelThing.tokFreq.getOrElse(w2.toLowerCase, 1).toFloat > 0.1
    if (rare1 && rare2 && w1.equalsIgnoreCase(w2))
      "Rare_Duplicate"
    else
      (if (rare1) "RARE" else w1) + "_AND_" + (if (rare2) "RARE" else w2)
  }

  val singPron = Set("i", "me", "my", "mine", "myself", "he", "she", "it", "him", "her", "his", "hers", "its", "one", "ones", "oneself", "this", "that")
  val pluPron = Set("we", "us", "our", "ours", "ourselves", "ourself", "they", "them", "their", "theirs", "themselves", "themself", "these", "those")
  val singDet = Set("a ", "an ", "this ")
  val pluDet = Set("those ", "these ", "some ")

  def numbersMatch(m1:Mention, m2:Mention): Char = {
    val n1 = m2.phrase.attr[Number].intValue
    val n2 = m1.phrase.attr[Number].intValue
    import NumberDomain._
    if (n1 == n2 && n1 != UNKNOWN)
      't'
    else if (n1 != n2 && n1 != UNKNOWN && n2 != UNKNOWN)
      'f'
    else if (n1 == UNKNOWN || n2 == UNKNOWN) {
      if (m1.phrase.toSeq.map(t => t.string.trim).mkString(" ").equals(m2.phrase.toSeq.map(t => t.string.trim).mkString(" ")))
        't'
      else
        'u'
    }
    else
      'u'
  }

  val relativizers = Set("who", "whom", "which", "whose", "whoever", "whomever", "whatever", "whichever", "that")

  def areAppositive(m1:Mention, m2:Mention): Boolean = {
    (m2.attr[MentionCharacteristics].isProper || m1.attr[MentionCharacteristics].isProper) &&
      (m2.phrase.last.next(2) == m1.phrase.head && m2.phrase.last.next.string.equals(",") ||
        m1.phrase.last.next(2) == m2.phrase.head && m1.phrase.last.next.string.equals(","))
  }

  def isRelativeFor(m1:Mention, m2:Mention) =
    relativizers.contains(m1.attr[MentionCharacteristics].lowerCaseHead) &&
      (m2.phrase.head == m1.phrase.last.next ||
        (m2.phrase.head == m1.phrase.last.next(2) && m1.phrase.last.next.string.equals(",")
          || m2.phrase.head == m1.phrase.last.next(2) && m1.phrase.last.next.string.equals(",")))


  def areRelative(m1:Mention, m2:Mention): Boolean = isRelativeFor(m1, m2) || isRelativeFor(m2, m1)

  def canBeAliases(m1:Mention, m2:Mention): Boolean = {
    val m1c = m1.attr[MentionCharacteristics]
    val m2c = m2.attr[MentionCharacteristics]
    val eType1 = m2c.predictEntityType
    val eType2 = m1c.predictEntityType

    val m1head = m2.phrase
    val m2head = m1.phrase
    val m1Words = m1head.phrase.split("\\s")
    val m2Words = m2head.phrase.split("\\s")

    if (m2c.isProper && m1c.isProper && m2c.predictEntityType.equals(m1c.predictEntityType) && (m2c.predictEntityType.equals("PERSON") || m2c.predictEntityType.equals("GPE")))
      return m2.phrase.last.string.toLowerCase equals m1.phrase.last.string.toLowerCase

    else if ((eType1.equals("ORG") || eType1.equals("unknown")) && (eType2.equals("ORG") || eType2.equals("unknown"))) {
      val (initials, shorter) =
        if (m1Words.length < m2Words.length)
          (m2c.initials, m1head.phrase)
        else
          (m1c.initials, m2head.phrase)
      return shorter.replaceAll("[., ]", "") equalsIgnoreCase initials
    }

    false
  }


  lazy val punct = "^['\"(),;.`]*(.*?)['\"(),;.`]*$".r
  def removePunct(s: String): String = {
    val punct(ret) = s
    ret
  }

}
