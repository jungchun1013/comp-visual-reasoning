# CLEVR Question Families

90 families (0-89), organized by 5 categories. Tags: `[spatial]` = uses `relate`, `[same_attr]` = uses `same_*`, `[intersect]` = uses `intersect`.

## Table of Contents

1. [Attribute Query](#attribute-query-32-families-53734-questions) (32 families, 53,734 questions)
   - [Direct filter](#direct-filter-4-families-depth-4-6)v
   - [Same-attribute](#same-attribute-12-families-depth-6-10-same_attr)v
   - [Spatial](#spatial-12-families-depth-7-18-spatial)v
   - [Spatial + intersect](#spatial--intersect-4-families-depth-13-16-spatial-intersect)
2. [Existence](#existence-12-families-20196-questions) (12 families, 20,196 questions)
   - [Direct filter](#direct-filter-1-family)v
   - [Same-attribute](#same-attribute-8-families-depth-5-10-same_attr)v
   - [Spatial](#spatial-3-families-depth-8-14-spatial)v
3. [Counting](#counting-21-families-35422-questions) (21 families, 35,422 questions)
   - [Direct filter](#direct-filter-1-family-1)v
   - [Same-attribute](#same-attribute-8-families-depth-6-8-same_attr)v
   - [Union / set operations](#union--set-operations-6-families-depth-8-10)
   - [Spatial](#spatial-5-families-depth-8-17-spatial)
   - [Spatial + intersect](#spatial--intersect-2-families-spatial-intersect)
4. [Attribute Comparison](#attribute-comparison-16-families-27098-questions) (16 families, 27,098 questions)
   - [Direct comparison](#direct-comparison-4-families-depth-11-12)v
   - [Spatial comparison](#spatial-comparison-12-families-depth-14-23-spatial)v
5. [Integer Comparison](#integer-comparison-9-families-13541-questions) (9 families, 13,541 questions)
   - [Direct comparison](#direct-comparison-3-families-depth-8-10)v
   - [Spatial comparison](#spatial-comparison-6-families-depth-13-20-spatial)v
6. [Summary](#summary)

---

## Attribute Query (32 families, 53,734 questions)

Final operation: `query_color`, `query_shape`, `query_material`, `query_size`

### Direct filter (4 families, depth 4-6)

- **Family 86** | depth=4
  - `scene → filter → unique → query_shape`
  - *"What shape is the brown thing?"* → cylinder
- **Family 87** | depth=5
  - `scene → filter → unique → query_material`
  - *"What is the material of the big purple object?"* → metal
- **Family 88** | depth=6
  - `scene → filter → unique → query_color`
  - *"What is the color of the large shiny sphere?"* → purple
- **Family 89** | depth=6
  - `scene → filter×3 → unique → query_size`
  - *"What size is the purple rubber sphere?"* → small

### Same-attribute (12 families, depth 6-10) `[same_attr]`

- **Family 53** | depth=6
  - `filter → unique → same_size → unique → query_material`
  - *"What is the material of the other object that is the same size as the matte thing?"* → metal
- **Family 59** | depth=6
  - `filter → unique → same_material → unique → query_color`
  - *"There is another thing that is the same material as the gray object; what is its color?"* → yellow
- **Family 55** | depth=7
  - `filter×2 → unique → same_color → unique → query_size`
  - *"The object that is the same color as the tiny cube is what size?"* → large
- **Family 57** | depth=7
  - `filter×2 → unique → same_color → unique → query_shape`
  - *"The other object that is the same color as the large shiny thing is what shape?"* → cylinder
- **Family 61** | depth=7
  - `filter×2 → unique → same_shape → unique → query_size`
  - *"There is another thing that is the same shape as the brown metallic object; what is its size?"* → small
- **Family 60** | depth=8
  - `filter×2 → unique → same_material → filter → unique → query_shape`
  - *"What is the shape of the large object that is made of the same material as the purple ball?"* → sphere
- **Family 52** | depth=9
  - `filter×2 → unique → same_size → filter×2 → unique → query_color`
  - *"What color is the matte ball that is the same size as the gray metal thing?"* → yellow
- **Family 54** | depth=9
  - `filter×2 → unique → same_size → filter×2 → unique → query_shape`
  - *"There is a gray rubber thing that is the same size as the gray sphere; what shape is it?"* → cube
- **Family 56** | depth=9
  - `filter×3 → unique → same_color → filter → unique → query_material`
  - *"What material is the sphere that is the same color as the small metal block?"* → metal
- **Family 58** | depth=9
  - `filter×2 → unique → same_material → filter×2 → unique → query_size`
  - *"What size is the brown block that is the same material as the large green object?"* → small
- **Family 63** | depth=9
  - `filter×2 → unique → same_shape → filter×2 → unique → query_material`
  - *"There is a big gray object that is the same shape as the purple rubber object; what is it made of?"* → rubber
- **Family 62** | depth=10
  - `filter×3 → unique → same_shape → filter×2 → unique → query_color`
  - *"The other small shiny thing that is the same shape as the tiny yellow shiny object is what color?"* → cyan

### Spatial (12 families, depth 7-18) `[spatial]`

- **Family 76** | depth=7
  - `filter×2 → unique → relate → unique → query_material`
  - *"What is the thing in front of the small metallic object made of?"* → rubber
- **Family 74** | depth=8
  - `filter×2 → unique → relate → filter → unique → query_size`
  - *"What size is the metallic thing that is left of the tiny blue thing?"* → large
- **Family 75** | depth=8
  - `filter×2 → unique → relate → filter → unique → query_color`
  - *"What color is the matte thing in front of the large cube?"* → cyan
- **Family 77** | depth=9
  - `filter×2 → unique → relate → filter×2 → unique → query_shape`
  - *"There is a big metallic thing left of the tiny green object; what is its shape?"* → sphere
- **Family 80** | depth=11
  - `filter×2 → unique → relate → filter×2 → unique → relate → unique → query_size`
  - *"There is a object in front of the metal cube that is to the right of the large cylinder; how big is it?"* → small
- **Family 81** | depth=12
  - `filter×3 → unique → relate → filter → unique → relate → filter → unique → query_color`
  - *"The cylinder that is to the right of the small object behind the tiny rubber cylinder is what color?"* → red
- **Family 82** | depth=12
  - `filter×3 → unique → relate → filter → unique → relate → filter → unique → query_material`
  - *"There is a yellow thing to the right of the rubber thing on the left side of the gray rubber cylinder; what is its material?"* → metal
- **Family 83** | depth=16
  - `filter×2 → unique → relate → filter×4 → unique → relate → filter×3 → unique → query_shape`
  - *"What is the shape of the small yellow rubber thing that is in front of the large yellow metal ball that is behind the small matte object?"* → cube
- **Family 27** | depth=15
  - `filter×3 → unique → relate → unique → relate → filter → unique → relate → filter×2 → unique → query_size`
  - *"What size is the yellow ball behind the sphere that is on the right side of the object that is behind the tiny yellow matte thing?"* → large
- **Family 30** | depth=15
  - `filter×3 → unique → relate → filter → unique → relate → filter → unique → relate → filter → unique → query_material`
  - *"What is the big thing that is in front of the block that is behind the block that is in front of the large shiny block made of?"* → rubber
- **Family 28** | depth=18
  - `filter×3 → unique → relate → filter → unique → relate → filter×3 → unique → relate → filter×2 → unique → query_shape`
  - *"What shape is the brown rubber object that is in front of the brown rubber block on the right side of the matte object that is on the left side of the tiny rubber cylinder?"* → cube
- **Family 29** | depth=18
  - `filter×3 → unique → relate → filter×2 → unique → relate → filter×2 → unique → relate → filter×2 → unique → query_color`
  - *"There is a tiny shiny object that is behind the big ball that is to the right of the big metallic thing behind the big brown cube; what is its color?"* → brown

### Spatial + intersect (4 families, depth 13-16) `[spatial, intersect]`

- **Family 35** | depth=13
  - `filter×2 → unique → relate → [filter×2 → unique → relate] → intersect → unique → query_shape`
  - *"There is a thing that is both to the left of the gray sphere and to the right of the small cylinder; what shape is it?"* → cube
- **Family 32** | depth=15
  - `filter×3 → unique → relate → [filter×2 → unique → relate] → intersect → filter → unique → query_size`
  - *"There is a metallic object that is left of the brown ball and in front of the tiny blue block; what is its size?"* → small
- **Family 34** | depth=15
  - `filter×4 → unique → relate → [filter×2 → unique → relate] → intersect → unique → query_material`
  - *"What is the material of the thing that is left of the blue block and on the right side of the big green matte block?"* → rubber
- **Family 33** | depth=16
  - `filter×4 → unique → relate → [filter×2 → unique → relate] → intersect → filter → unique → query_color`
  - *"The large thing that is both on the left side of the purple shiny object and behind the tiny gray metallic ball is what color?"* → brown

---

## Existence (12 families, 20,196 questions)

Final operation: `exist`

### Direct filter (1 family)

- **Family 85** | depth=5
  - `scene → filter×3 → exist`
  - *"Are any tiny green metal things visible?"* → no

### Same-attribute (8 families, depth 5-10) `[same_attr]`

- **Family 36** | depth=5
  - `filter → unique → same_size → exist`
  - *"Are there any other things that are the same size as the brown object?"* → yes
- **Family 38** | depth=5
  - `filter → unique → same_material → exist`
  - *"Is there anything else that has the same material as the red thing?"* → yes
- **Family 39** | depth=6
  - `filter×2 → unique → same_shape → exist`
  - *"Are there any other things that are the same shape as the big metallic object?"* → no
- **Family 37** | depth=7
  - `filter×3 → unique → same_color → exist`
  - *"Is there anything else that has the same color as the large shiny cube?"* → yes
- **Family 47** | depth=7
  - `filter → unique → same_shape → filter×2 → exist`
  - *"Is there a big brown object of the same shape as the green thing?"* → yes
- **Family 44** | depth=8
  - `filter×2 → unique → same_size → filter×2 → exist`
  - *"Is there a blue metal object that has the same size as the gray metal object?"* → yes
- **Family 46** | depth=9
  - `filter×3 → unique → same_material → filter×2 → exist`
  - *"Is there a small green thing that has the same material as the large brown ball?"* → no
- **Family 45** | depth=10
  - `filter×3 → unique → same_color → filter×3 → exist`
  - *"Are there any large matte blocks of the same color as the large metal ball?"* → yes

### Spatial (3 families, depth 8-14) `[spatial]`

- **Family 73** | depth=8
  - `filter×3 → unique → relate → filter → exist`
  - *"There is a small gray block; are there any spheres to the left of it?"* → yes
- **Family 26** | depth=14
  - 3 relate hops → filter → exist
  - *"Is there a metallic object left of the gray object that is behind the large cylinder that is in front of the green matte object?"* → no
- **Family 79** | depth=14
  - 2 relate hops → filter×3 → exist
  - *"Are there any gray rubber spheres on the left side of the matte block that is to the right of the tiny rubber cylinder?"* → no

---

## Counting (21 families, 35,422 questions)

Final operation: `count`

### Direct filter (1 family)

- **Family 84** | depth=5
  - `scene → filter×3 → count`
  - *"What number of gray matte blocks are there?"* → 1

### Same-attribute (8 families, depth 6-8) `[same_attr]`

- **Family 40** | depth=6
  - `filter×2 → unique → same_size → count`
  - *"What number of other objects are the same size as the purple shiny object?"* → 2
- **Family 43** | depth=6
  - `filter×2 → unique → same_shape → count`
  - *"What number of other things are there of the same shape as the tiny matte object?"* → 2
- **Family 41** | depth=6
  - `filter×2 → unique → same_color → count`
  - *"How many other objects are there of the same color as the matte cylinder?"* → 1
- **Family 42** | depth=7
  - `filter×3 → unique → same_material → count`
  - *"What number of other things are the same material as the big gray cylinder?"* → 6
- **Family 48** | depth=7
  - `filter×2 → unique → same_size → filter → count`
  - *"How many red objects have the same size as the red cylinder?"* → 1
- **Family 49** | depth=7
  - `filter → unique → same_color → filter×2 → count`
  - *"What number of small shiny objects have the same color as the block?"* → 1
- **Family 50** | depth=8
  - `filter×2 → unique → same_material → filter×2 → count`
  - *"What number of other small balls have the same material as the yellow ball?"* → 0
- **Family 51** | depth=8
  - `filter×2 → unique → same_shape → filter×2 → count`
  - *"How many small metallic things are the same shape as the big brown thing?"* → 1

### Union / set operations (6 families, depth 8-10)

- **Family 64** | depth=8
  - `filter_size → [filter_size → filter_color → filter_shape] → union → count`
  - *"What number of objects are big objects or tiny yellow balls?"* → 4
- **Family 68** | depth=9
  - `filter_size → filter_color → filter_shape → [filter_color] → union → filter_material → count`
  - *"How many metallic objects are big blue cubes or blue objects?"* → 2
- **Family 69** | depth=9
  - `[filter_color → filter_size → filter_color → filter_material] → union → filter_shape → count`
  - *"What number of cylinders are gray objects or tiny brown matte objects?"* → 1
- **Family 65** | depth=10
  - `filter_color → filter_material → filter_shape → [filter_color → filter_shape] → union → filter_size → count`
  - *"How many large things are either cyan metallic cylinders or yellow blocks?"* → 0
- **Family 66** | depth=10
  - `filter_size → filter_material → filter_shape → [filter_size → filter_material] → union → filter_color → count`
  - *"What number of yellow objects are big matte cubes or small metal objects?"* → 1

### Spatial (5 families, depth 8-17) `[spatial]`

- **Family 72** | depth=8
  - `filter×2 → unique → relate → filter×2 → count`
  - *"How many tiny matte objects are to the left of the small yellow thing?"* → 0
- **Family 71** | depth=12
  - `filter_size → filter_shape → [filter_color → filter_material → unique → relate → filter_color → filter_shape] → union → count`
  - *"What number of objects are tiny spheres or brown blocks behind the gray matte object?"* → 2
- **Family 67** | depth=14
  - `filter×4 → unique → relate → [filter×2 → unique → relate] → union → count`
  - *"What number of things are objects behind the big green matte cube or things that are in front of the big shiny thing?"* → 3
- **Family 70** | depth=14
  - `filter×4 → unique → relate → filter_material → [filter×3] → union → count`
  - *"How many objects are either metal things behind the small green rubber cylinder or small green rubber objects?"* → 2
- **Family 78** | depth=14
  - `filter×4 → unique → relate → filter → unique → relate → filter×3 → count`
  - *"How many tiny yellow matte things are to the right of the purple thing in front of the small cyan shiny cube?"* → 0

### Spatial + intersect (2 families) `[spatial, intersect]`

- **Family 25** | depth=17
  - `filter×4 → unique → relate → filter → unique → relate → filter → unique → relate → filter×3 → count`
  - *"There is a cube to the left of the rubber thing that is on the right side of the large green matte block; how many big blue metallic objects are right of it?"* → 1
- **Family 31** | depth=15
  - `filter×3 → unique → relate → [filter×3 → unique → relate] → intersect → filter → count`
  - *"What number of tiny things are both on the left side of the gray shiny sphere and to the right of the brown rubber cube?"* → 1

---

## Attribute Comparison (16 families, 27,098 questions)

Final operation: `equal_size`, `equal_color`, `equal_material`, `equal_shape`

### Direct comparison (4 families, depth 11-12)

- **Family 9** | depth=11 | `equal_size`
  - `[filter×2 → unique → query_size] → [filter×2 → unique → query_size] → equal_size`
  - *"Do the purple cylinder and the yellow rubber thing have the same size?"* → no
- **Family 10** | depth=12 | `equal_color`
  - `[filter×2 → unique → query_color] → [filter×3 → unique → query_color] → equal_color`
  - *"Is the color of the big matte object the same as the large metal cube?"* → yes
- **Family 11** | depth=11 | `equal_material`
  - `[filter×2 → unique → query_material] → [filter×2 → unique → query_material] → equal_material`
  - *"Is the material of the yellow block the same as the yellow cylinder?"* → no
- **Family 12** | depth=11 | `equal_shape`
  - `[filter → unique → query_shape] → [filter×3 → unique → query_shape] → equal_shape`
  - *"Is the purple thing the same shape as the large gray rubber thing?"* → no

### Spatial comparison (12 families, depth 14-23) `[spatial]`

- **Family 13** | depth=15 | `equal_size`
  - `[filter×2 → unique → relate → filter → unique → query_size] → [filter×3 → unique → query_size] → equal_size`
  - *"There is a rubber object that is left of the yellow block; is it the same size as the tiny rubber block?"* → no
- **Family 14** | depth=16 | `equal_size`
  - `[filter×2 → unique → query_size] → [filter×3 → unique → relate → filter×2 → unique → query_size] → equal_size`
  - *"There is a matte sphere; is it the same size as the yellow ball behind the tiny metal cylinder?"* → no
- **Family 15** | depth=18 | `equal_size`
  - `[filter×2 → unique → relate → unique → query_size] → [filter×3 → unique → relate → filter×2 → unique → query_size] → equal_size`
  - *"There is a thing that is behind the small gray object; is its size the same as the matte block in front of the small brown block?"* → no
- **Family 16** | depth=16 | `equal_color`
  - `[filter×2 → unique → relate → filter×2 → unique → query_color] → [filter×3 → unique → query_color] → equal_color`
  - *"Is the color of the metal block that is right of the yellow rubber object the same as the large metal cylinder?"* → yes
- **Family 17** | depth=16 | `equal_color`
  - `[filter → unique → query_color] → [filter×3 → unique → relate → filter×3 → unique → query_color] → equal_color`
  - *"There is a cylinder; does it have the same color as the big metallic cube on the left side of the small metallic block?"* → no
- **Family 18** | depth=15 | `equal_color`
  - `[filter×2 → unique → relate → unique → query_color] → [filter → unique → relate → filter → unique → query_color] → equal_color`
  - *"Is the color of the object to the left of the small sphere the same as the cylinder right of the purple thing?"* → no
- **Family 19** | depth=16 | `equal_material`
  - `[filter×3 → unique → relate → filter×2 → unique → query_material] → [filter×2 → unique → query_material] → equal_material`
  - *"Do the purple cylinder that is behind the small brown cube and the small yellow thing have the same material?"* → yes
- **Family 20** | depth=15 | `equal_material`
  - `[filter×3 → unique → query_material] → [filter×2 → unique → relate → filter → unique → query_material] → equal_material`
  - *"Does the large yellow sphere have the same material as the sphere in front of the tiny blue object?"* → yes
- **Family 21** | depth=23 | `equal_material`
  - `[filter×4 → unique → relate → filter×3 → unique → query_material] → [filter×4 → unique → relate → filter → unique → query_material] → equal_material`
  - *"Do the large yellow ball behind the small yellow matte cube and the big object that is in front of the big gray shiny sphere have the same material?"* → no
- **Family 22** | depth=16 | `equal_shape`
  - `[filter×3 → unique → relate → filter×2 → unique → query_shape] → [filter×2 → unique → query_shape] → equal_shape`
  - *"There is a large green object on the left side of the small purple block; does it have the same shape as the purple rubber object?"* → no
- **Family 23** | depth=14 | `equal_shape`
  - `[filter×2 → unique → query_shape] → [filter×2 → unique → relate → filter → unique → query_shape] → equal_shape`
  - *"Does the green rubber object have the same shape as the gray thing that is on the right side of the big purple object?"* → no
- **Family 24** | depth=21 | `equal_shape`
  - `[filter×4 → unique → relate → filter×2 → unique → query_shape] → [filter×2 → unique → relate → filter×2 → unique → query_shape] → equal_shape`
  - *"Does the brown metal thing on the left side of the large red rubber cube have the same shape as the big metallic thing right of the brown cylinder?"* → yes

---

## Integer Comparison (9 families, 13,541 questions)

Final operation: `equal_integer`, `less_than`, `greater_than`

### Direct comparison (3 families, depth 8-10)

- **Family 0** | depth=8 | `equal_integer`
  - `[filter×2 → count] → [filter → count] → equal_integer`
  - *"Are there the same number of tiny blocks and matte objects?"* → no
- **Family 1** | depth=10 | `less_than`
  - `[filter×3 → count] → [filter×2 → count] → less_than`
  - *"Are there fewer tiny shiny cubes than green metallic things?"* → yes
- **Family 2** | depth=9 | `greater_than`
  - `[filter → count] → [filter×3 → count] → greater_than`
  - *"Are there more green objects than tiny rubber cylinders?"* → yes

### Spatial comparison (6 families, depth 13-20) `[spatial]`

- **Family 3** | depth=13 | `equal_integer`
  - `[filter → unique → relate → filter → count] → [filter×4 → count] → equal_integer`
  - *"Are there an equal number of small things that are left of the gray object and large cyan matte balls?"* → no
- **Family 4** | depth=15 | `less_than`
  - `[filter×3 → unique → relate → filter×2 → count] → [filter×3 → count] → less_than`
  - *"Is the number of brown cylinders in front of the brown matte cylinder less than the number of brown rubber cylinders?"* → no
- **Family 5** | depth=13 | `greater_than`
  - `[filter×2 → unique → relate → filter → count] → [filter×3 → count] → greater_than`
  - *"Are there more shiny things that are to the left of the yellow rubber thing than small red spheres?"* → yes
- **Family 6** | depth=19 | `equal_integer`
  - `[filter×4 → unique → relate → filter×2 → count] → [filter×2 → unique → relate → filter×2 → count] → equal_integer`
  - *"Is the number of big brown objects behind the tiny red shiny cylinder the same as the number of metallic blocks that are on the right side of the gray ball?"* → no
- **Family 7** | depth=20 | `less_than`
  - `[filter×2 → unique → relate → filter×3 → count] → [filter×2 → unique → relate → filter×4 → count] → less_than`
  - *"Are there fewer large purple spheres that are to the right of the big blue thing than tiny brown metal cylinders that are left of the matte cylinder?"* → yes
- **Family 8** | depth=18 | `greater_than`
  - `[filter×3 → unique → relate → filter → count] → [filter×3 → unique → relate → filter×2 → count] → greater_than`
  - *"Is the number of gray objects in front of the large yellow metal object greater than the number of yellow spheres to the right of the small metal sphere?"* → yes

---

## Summary

- **Attribute Query**: 32 families, 53,734 questions — direct (4), same_attr (12), spatial (12), intersect (4)
- **Existence**: 12 families, 20,196 questions — direct (1), same_attr (8), spatial (3)
- **Counting**: 21 families, 35,422 questions — direct (1), same_attr (8), union (6), spatial (5), intersect (2)
- **Attribute Comparison**: 16 families, 27,098 questions — direct (4), spatial (12)
- **Integer Comparison**: 9 families, 13,541 questions — direct (3), spatial (6)
- **Total**: 90 families, 149,991 questions

### Building blocks

- `filter_X(v)` — Keep objects where attribute X = v
- `unique` — Assert exactly one object remains
- `relate(dir)` — Find objects in spatial relation (left/right/front/behind)
- `same_X` — Find objects sharing attribute X with anchor
- `union` / `intersect` — Set operations on object sets
- `query_X` — Return attribute X of the object
- `equal_X` — Compare attribute X of two objects
- `count` — Count remaining objects
- `exist` — Check if any objects remain (yes/no)
- `less_than` / `greater_than` — Compare two counts
