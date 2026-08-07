| Sensitivity family    | Analysis                                         |   ARI vs final |   Silhouette |   Minimum cluster size |   Maximum cluster size |
|:----------------------|:-------------------------------------------------|---------------:|-------------:|-----------------------:|-----------------------:|
| Core specification    | BASELINE_RECOMPUTED_PCA85_HAC_K2                 |         1      |       0.4108 |                     13 |                     68 |
| Core specification    | NO_PCA_STANDARDIZED_HAC_K2                       |         1      |       0.3603 |                     13 |                     68 |
| Core specification    | RAW_0_1_NO_STANDARDIZATION_HAC_K2                |         1      |       0.3565 |                     13 |                     68 |
| Core specification    | AUGMENTED_15_PCA85_HAC_K2                        |         0.9365 |       0.3934 |                     12 |                     69 |
| Core specification    | AUGMENTED_15_NO_PCA_HAC_K2                       |         0.9365 |       0.3486 |                     12 |                     69 |
| Leave-one-feature-out | Remove admin_time_load_constraint                |         0.9389 |       0.4113 |                     14 |                     67 |
| Leave-one-feature-out | Remove willingness_constraint_score              |         1      |       0.4191 |                     13 |                     68 |
| Leave-one-feature-out | Remove training_deficit_score                    |         1      |       0.441  |                     13 |                     68 |
| Leave-one-feature-out | Remove digital_usage_constraint_score            |         1      |       0.3929 |                     13 |                     68 |
| Leave-one-feature-out | Remove digital_tool_variety_constraint           |         1      |       0.3878 |                     13 |                     68 |
| Leave-one-feature-out | Remove internet_stability_constraint             |         1      |       0.3872 |                     13 |                     68 |
| Leave-one-feature-out | Remove staffing_constraint_score                 |         1      |       0.4212 |                     13 |                     68 |
| Leave-one-feature-out | Remove device_constraint                         |         1      |       0.388  |                     13 |                     68 |
| Leave-one-feature-out | Remove time_constraint_score                     |         1      |       0.4225 |                     13 |                     68 |
| Leave-one-feature-out | Remove administrative_disorganization_constraint |         1      |       0.4258 |                     13 |                     68 |
| Leave-one-feature-out | Remove recording_system_constraint               |         1      |       0.4003 |                     13 |                     68 |
| Leave-one-feature-out | Remove system_change_resistance_constraint       |         1      |       0.4074 |                     13 |                     68 |
| Leave-one-feature-out | Remove resource_constraint_score                 |         1      |       0.4017 |                     13 |                     68 |
| PCA threshold         | 80% retained variance                            |         1      |       0.4276 |                     13 |                     68 |
| PCA threshold         | 85% retained variance                            |         1      |       0.4108 |                     13 |                     68 |
| PCA threshold         | 90% retained variance                            |         1      |       0.3961 |                     13 |                     68 |
| PCA threshold         | 95% retained variance                            |         1      |       0.3765 |                     13 |                     68 |