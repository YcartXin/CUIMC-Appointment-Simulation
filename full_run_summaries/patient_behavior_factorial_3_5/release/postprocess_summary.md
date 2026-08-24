# Patient behavior factorial postprocessing

Backgrounds: 540
Clinic contexts: 60
Behavior cells: 9
Evaluation seeds per selected cell: 10
Bootstrap draws: 2000

The factorial contrasts operate on paired policy effects.

For example:
effect_noshow = (
    policy effect under higher no-show severity
    - policy effect under lower no-show severity
)

The interaction rows are difference-in-differences across
the no-show and balking dimensions.
