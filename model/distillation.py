# Importing the specialized trainer for model distillation from the setfit package
from setfit import DistillationSetFitTrainer

# Define a function to perform model distillation using a student model and a teacher model
def perform_model_distillation(learner_model, instructor_model, training_data):
    """
    Initializes the distillation process where the learner (student) model learns from the
    instructor (teacher) model using the provided training data.

    Parameters:
    - learner_model: The student model that will be trained.
    - instructor_model: The teacher model that provides guidance.
    - training_data: Dataset used for training the student model.

    Returns:
    - An instance of DistillationSetFitTrainer after training.
    """

    # Create a distillation trainer with the specified models and dataset
    distillation_trainer = DistillationSetFitTrainer(
        teacher_model=instructor_model,
        train_dataset=training_data,
        student_model=learner_model,
    )

    # Execute the training process for the student model
    distillation_trainer.train()

    # Return the distillation trainer instance
    return distillation_trainer

"""
DistillationSetFitTrainer is a specialized training class provided by the SetFit library, 
designed specifically for the process of model distillation. Model distillation is a technique
where a smaller, less complex "student" model learns to replicate the performance of a larger, 
more complex "teacher" model. This approach is commonly used to create models that are more efficient 
and faster at inference while retaining much of the performance of the original model.

Typically, distillation involves a combination of a traditional loss function (like cross-entropy) 
to measure the accuracy of the student model on the ground truth labels, and a distillation loss, 
which measures how well the student model's outputs match those of the teacher model. 
Common choice is the Kullback-Leibler divergence for the distillation loss.

In the distillation process, temperature scaling is often used during the computation of the softmax function 
to smooth out the probabilities and make the outputs from the teacher model softer. 
This helps the student model to learn more generalized features rather than mimicking hard probabilities.

"""