/* Copyright (C) 2008-2010 University of Massachusetts Amherst,
   Department of Computer Science.
   This file is part of "FACTORIE" (Factor graphs, Imperative, Extensible)
   http://factorie.cs.umass.edu, http://code.google.com/p/factorie/
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

package cc.factorie.la
import cc.factorie._
import cc.factorie.util._

trait Tensor4 extends Tensor {
  def dim1: Int
  def dim2: Int
  def dim3: Int
  def dim4: Int
  def activeDomain1: IntSeq
  def activeDomain2: IntSeq
  def activeDomain3: IntSeq
  def activeDomain4: IntSeq
  def numDimensions: Int = 4
  def activeDomains = Array(activeDomain1, activeDomain2)
  def dimensions = Array(dim1, dim2, dim3, dim4)
  def apply(i:Int, j:Int, k:Int, l:Int): Double = apply(i*dim2*dim3*dim4 + j*dim2*dim3 + k*dim3 + l)
  def update(i:Int, j:Int, k:Int, l:Int, v:Double): Unit = update(i*dim2*dim3*dim4 + j*dim2*dim3 + k*dim3 + l, v)
  def +=(i:Int, j:Int, k:Int, l:Int, v:Double): Unit = +=(singleIndex(i, j, k, l), v)
  @inline final def length = dim1 * dim2 * dim3 * dim4
  @inline final def singleIndex(i:Int, j:Int, k:Int, l:Int): Int = i*dim2*dim3*dim4 + j*dim2*dim3 + k*dim3 + l 
  @inline final def multiIndex(i:Int): (Int, Int, Int, Int) = (i/dim2/dim3/dim4, (i/dim3/dim4)%dim2, (i/dim4)%dim3, i%dim4)
}

trait DenseTensorLike4 extends Tensor4 with DenseTensorLike {
  private var __values = new Array[Double](dim1*dim2*dim3*dim4)
  protected def _values = __values
  def isDense = true
  def activeDomain1 = new RangeIntSeq(0, dim1)
  def activeDomain2 = new RangeIntSeq(0, dim2)
  def activeDomain3 = new RangeIntSeq(0, dim3)
  def activeDomain4 = new RangeIntSeq(0, dim4)
  def activeDomain = new RangeIntSeq(0, dim1*dim2*dim3*dim4)
  def apply(i:Int): Double = __values(i)
  override def apply(i:Int, j:Int, k:Int, l:Int): Double = __values(i*dim2*dim3*dim4 + j*dim2*dim3 + k*dim3 + l)
  override def update(i:Int, v:Double): Unit = __values(i) = v
  override def update(i:Int, j:Int, k:Int, l:Int, v:Double): Unit = __values(i*dim2*dim3*dim4 + j*dim2*dim3 + k*dim3 + l) = v
  override def +=(i:Int, v:Double): Unit = __values(i) += v
}
class DenseTensor4(val dim1:Int, val dim2:Int, val dim3:Int, val dim4:Int) extends DenseTensorLike4

class SingletonBinaryTensor4(val dim1:Int, val dim2:Int, val dim3:Int, val dim4:Int, val singleIndex1:Int, val singleIndex2:Int, val singleIndex3:Int, val singleIndex4:Int) extends Tensor4 with SingletonBinaryTensor {
  def activeDomain1 = new SingletonIntSeq(singleIndex1)
  def activeDomain2 = new SingletonIntSeq(singleIndex2)
  def activeDomain3 = new SingletonIntSeq(singleIndex3)
  def activeDomain4 = new SingletonIntSeq(singleIndex4)
  val singleIndex = singleIndex1*dim2*dim3*dim4 + singleIndex2*dim3*dim4 + singleIndex3*dim4 + singleIndex4
}

class SingletonTensor4(dim1:Int, dim2:Int, dim3:Int, dim4:Int, singleIndex1:Int, singleIndex2:Int, singleIndex3:Int, singleIndex4:Int, val singleValue:Double) extends SingletonBinaryTensor4(dim1, dim2, dim3, dim4, singleIndex1, singleIndex2, singleIndex3, singleIndex4) with SingletonTensor






trait Dense3LayeredTensorLike4 extends Tensor4 {
  def newTensor1(dim:Int): Tensor1
  private var _inners = Array.fill(dim1*dim2*dim3)(newTensor1(dim3))
  override def apply(i:Int, j:Int, k:Int, l:Int): Double = _inners(i*dim2*dim3 + j*dim2 + k).apply(l)
  def apply(i:Int): Double = apply(i/dim2/dim3/dim4, (i/dim3/dim4)%dim2, (i/dim4)%dim3, i%dim4)
  override def update(i:Int, j:Int, k:Int, l:Int, v:Double): Unit = _inners(i*dim2*dim3 + j*dim2 + k).update(l, v)
}
abstract class Dense3LayeredTensor4(val dim1:Int, val dim2:Int, val dim3:Int) extends Dense3LayeredTensorLike4

trait Singleton3LayeredTensorLike4 extends Tensor4 {
  def singleIndex1: Int
  def singleIndex2: Int
  def singleIndex3: Int
  def inner: Tensor1
  def isDense = false
  def activeDomain1 = new SingletonIntSeq(singleIndex1)
  def activeDomain2 = new SingletonIntSeq(singleIndex2)
  def activeDomain3 = new SingletonIntSeq(singleIndex3)
  def activeDomain4 = inner.activeDomain1
  def activeDomain = { val offset = singleIndex1*dim2*dim3*dim4 + singleIndex2*dim3*dim4 + singleIndex3*dim4; inner.activeDomain1.map(_ * offset) }
  override def apply(i:Int, j:Int, k:Int, l:Int): Double = if (i == singleIndex1 && j == singleIndex2 && k == singleIndex3) inner.apply(l) else 0.0
  def apply(i:Int): Double = apply(i/dim2/dim3/dim4, (i/dim3/dim4)%dim2, (i/dim4)%dim3, i%dim4)
  override def update(i:Int, j:Int, k:Int, l:Int, v:Double): Unit = if (i == singleIndex1 && j == singleIndex2 && k == singleIndex3) inner.update(l, v) else throw new Error("Outer indices out of bounds: "+List(i,j,k))
}
class Singleton3LayeredTensor4(val dim1:Int, val dim2:Int, val dim3:Int, val dim4:Int, val singleIndex1:Int, val singleIndex2:Int, val singleIndex3:Int, val inner:Tensor1) extends Singleton3LayeredTensorLike4
