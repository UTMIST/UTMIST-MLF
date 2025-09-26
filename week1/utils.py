"""
Utility functions for grading machine learning exercises.
Similar to Andrew Ng's Coursera lab grading system.
"""

import torch
import numpy as np


def grade_gradient_w(error, hours, student_answer, tolerance=1e-6):
    """
    Grade the gradient calculation for weight parameter w.

    Args:
        error: prediction error tensor (y_pred - scores)
        hours: input hours tensor
        student_answer: student's computed grad_w
        tolerance: numerical tolerance for comparison

    Returns:
        dict: grading result with 'correct', 'message', and 'expected_answer'
    """
    # Correct answer: grad_w = 2/N * sum(error * hours)
    N = hours.numel()
    expected_grad_w = 2.0 * torch.sum(error * hours) / N

    # Check if student provided an answer
    if student_answer is None:
        return {
            'correct': False,
            'message': '❌ Please compute grad_w using the formula: grad_w = 2/N * sum(error * hours)',
            'expected_answer': expected_grad_w.item()
        }

    # Convert to tensor if needed
    if not isinstance(student_answer, torch.Tensor):
        student_answer = torch.tensor(student_answer, dtype=torch.float32)

    # Check if shapes match
    if student_answer.shape != expected_grad_w.shape:
        return {
            'correct': False,
            'message': f'❌ Shape mismatch. Expected shape {expected_grad_w.shape}, got {student_answer.shape}',
            'expected_answer': expected_grad_w.item()
        }

    # Check if values are close enough
    if torch.allclose(student_answer, expected_grad_w, atol=tolerance):
        return {
            'correct': True,
            'message': '✅ Correct! grad_w = 2/N * sum(error * hours)',
            'expected_answer': expected_grad_w.item()
        }
    else:
        return {
            'correct': False,
            'message': f'❌ Incorrect. Expected {expected_grad_w.item():.6f}, got {student_answer.item():.6f}',
            'expected_answer': expected_grad_w.item()
        }


def grade_gradient_b(error, student_answer, tolerance=1e-6):
    """
    Grade the gradient calculation for bias parameter b.

    Args:
        error: prediction error tensor (y_pred - scores)
        student_answer: student's computed grad_b
        tolerance: numerical tolerance for comparison

    Returns:
        dict: grading result with 'correct', 'message', and 'expected_answer'
    """
    # Correct answer: grad_b = 2/N * sum(error)
    N = error.numel()
    expected_grad_b = 2.0 * torch.sum(error) / N

    # Check if student provided an answer
    if student_answer is None:
        return {
            'correct': False,
            'message': '❌ Please compute grad_b using the formula: grad_b = 2/N * sum(error)',
            'expected_answer': expected_grad_b.item()
        }

    # Convert to tensor if needed
    if not isinstance(student_answer, torch.Tensor):
        student_answer = torch.tensor(student_answer, dtype=torch.float32)

    # Check if shapes match
    if student_answer.shape != expected_grad_b.shape:
        return {
            'correct': False,
            'message': f'❌ Shape mismatch. Expected shape {expected_grad_b.shape}, got {student_answer.shape}',
            'expected_answer': expected_grad_b.item()
        }

    # Check if values are close enough
    if torch.allclose(student_answer, expected_grad_b, atol=tolerance):
        return {
            'correct': True,
            'message': '✅ Correct! grad_b = 2/N * sum(error)',
            'expected_answer': expected_grad_b.item()
        }
    else:
        return {
            'correct': False,
            'message': f'❌ Incorrect. Expected {expected_grad_b.item():.6f}, got {student_answer.item():.6f}',
            'expected_answer': expected_grad_b.item()
        }


def grade_both_gradients(error, hours, grad_w_answer, grad_b_answer, tolerance=1e-6):
    """
    Grade both gradient calculations at once.

    Args:
        error: prediction error tensor (y_pred - scores)
        hours: input hours tensor
        grad_w_answer: student's computed grad_w
        grad_b_answer: student's computed grad_b
        tolerance: numerical tolerance for comparison

    Returns:
        dict: combined grading result
    """
    w_result = grade_gradient_w(error, hours, grad_w_answer, tolerance)
    b_result = grade_gradient_b(error, grad_b_answer, tolerance)

    both_correct = w_result['correct'] and b_result['correct']

    return {
        'correct': both_correct,
        'grad_w_result': w_result,
        'grad_b_result': b_result,
        'message': f"grad_w: {w_result['message']}\ngrad_b: {b_result['message']}"
    }


def print_grading_result(result):
    """
    Print grading result in a nice format.

    Args:
        result: grading result dictionary
    """
    print("=" * 50)
    print("GRADING RESULT")
    print("=" * 50)

    if 'grad_w_result' in result:  # Combined grading
        print(result['message'])
        print(
            f"\nOverall: {'✅ PASSED' if result['correct'] else '❌ NEEDS WORK'}")
    else:  # Single grading
        print(result['message'])
        print(
            f"\nStatus: {'✅ CORRECT' if result['correct'] else '❌ INCORRECT'}")

    print("=" * 50)


def run_gradient_test():
    """
    Run a simple test to verify the grading functions work correctly.
    """
    print("Testing grading functions...")

    # Create test data
    torch.manual_seed(42)
    error = torch.tensor([1.0, 2.0, 3.0])
    hours = torch.tensor([2.0, 4.0, 6.0])

    # Test correct answers
    correct_grad_w = 2.0 * torch.sum(error * hours) / hours.numel()
    correct_grad_b = 2.0 * torch.sum(error) / error.numel()

    print("\n1. Testing correct answers:")
    result = grade_both_gradients(error, hours, correct_grad_w, correct_grad_b)
    print_grading_result(result)

    print("\n2. Testing incorrect answers:")
    wrong_grad_w = torch.tensor(1.0)
    wrong_grad_b = torch.tensor(2.0)
    result = grade_both_gradients(error, hours, wrong_grad_w, wrong_grad_b)
    print_grading_result(result)

    print("\n3. Testing None answers:")
    result = grade_both_gradients(error, hours, None, None)
    print_grading_result(result)


if __name__ == "__main__":
    run_gradient_test()
