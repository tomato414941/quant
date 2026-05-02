# Predictor Feature Recipe Ownership

## Issue

`FeatureSpec` がranking-derived featureの定義だけでなく、ranking strategy由来の
feature construction recipeまで持っている。

現状は `FeatureSpec.ranking_feature_recipe` に `RankingFeatureRecipeSpec` を埋め込み、
`compute_predictor_panel()` がそこからselection/ranking情報を取り出してpredictor用の
feature panelを作る。

## Current State

Current dependency flow:

```text
PredictorSpec
  -> FeatureSpec
    -> ranking_feature_recipe
      -> selection/ranking recipe
```

This supports reproducible predictor/forecast research. A predictor run can
rebuild ranking-derived features from the recipe stored with the feature spec.

## Concern

`FeatureSpec` may be carrying too much responsibility:

- feature names and inputs
- derived feature declarations
- ranking strategy provenance
- enough information to reconstruct ranking-derived feature panels

This couples predictor research to selection/ranking internals. The issue is not
that ranking-derived features exist; the issue is that feature definition and
feature construction provenance are stored in the same object.

## Direction

Do not remove `ranking_feature_recipe` now. Predictor / forecast research still
needs reproducible feature construction.

If this area is revisited, consider moving ranking-derived feature construction
into a dedicated feature recipe or feature builder layer. `PredictorSpec` could
then reference that recipe instead of embedding selection/ranking construction
details directly inside `FeatureSpec`.
