

def test_formato_compatto_delle_citazioni():
    from src.answer_quality import clean_answer, inspect_answer
    assert clean_answer("Valore [2] e altro [#2, #5].") == "Valore [#2] e altro [#2] [#5]."
    assert inspect_answer("64 stabilimenti [#2].", 8)["status"] == "cited"


def test_il_prompt_non_propone_piu_citazioni_a_forma_di_link():
    from pathlib import Path
    prompt = (Path(__file__).resolve().parent.parent / "prompts" / "system_prompt.txt").read_text()
    assert "](#" not in prompt
    assert "[#2]" in prompt
