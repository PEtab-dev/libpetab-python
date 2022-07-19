"""Test related to petab.models.model_pysb"""
import pysb
from petab.models.pysb_model import PySBModel, parse_species_name


def test_parse_species_name():
    assert parse_species_name("cyclin(Y='U', b=None)") \
           == [("cyclin", None, {'Y': 'U', 'b': None})]

    assert parse_species_name("cdc2(Y='P', b=1) % cyclin(Y='P', b=1)") \
           == [("cdc2", None, {'Y': 'P', 'b': 1}),
               ("cyclin", None, {'Y': 'P', 'b': 1})]

    assert parse_species_name("A()") \
           == [("A", None, {})]

    assert parse_species_name(
        'Bax(s1=1, s2=2, t=None) % Bax(s1=3, s2=1, t=None) % '
        'Bax(s1=2, s2=3, t=None)') \
        == [("Bax", None, {'s1': 1, 's2': 2, 't': None}),
            ("Bax", None, {'s1': 3, 's2': 1, 't': None}),
            ("Bax", None, {'s1': 2, 's2': 3, 't': None})]

    assert parse_species_name('A(b=None) ** X') \
           == [("A", "X", {'b': None})]

    assert parse_species_name('A(b=1) ** X % B(a=1) ** X') \
           == [("A", "X", {'b': 1}),
               ("B", "X", {'a': 1})]

    # TODO: MultiState


def test_pysb_model():
    model = pysb.Model()
    model.add_component(pysb.Compartment("c1"))
    model.add_component(pysb.Monomer("A"))
    model.add_component(pysb.Monomer("B", ["s"], {'s': ["a", "b"]}))
    petab_model = PySBModel(model=model, model_id="test_model")

    assert petab_model.is_state_variable("A()") is True
    assert petab_model.is_state_variable("B(s='a')") is True

    # a compartment
    assert petab_model.is_state_variable("c1") is False
    # a monomer
    assert petab_model.is_state_variable("A") is False
    assert petab_model.is_state_variable("B") is False

    # not concrete
    assert petab_model.is_state_variable("B()") is False

    # non-existing compartment
    assert petab_model.is_state_variable("A() ** c2") is False

    # non-existing site
    assert petab_model.is_state_variable("A(s='a')") is False
