# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

from deepmd.pt.model.atomic_model import (
    DPAtomicModel,
    PairTabAtomicModel,
)
from deepmd.pt.model.descriptor import (
    DescrptDPA1,
    DescrptDPA2,
    DescrptDPA3,
    DescrptHybrid,
    DescrptSeA,
    DescrptSeR,
    DescrptSeT,
    DescrptSeTTebd,
)
from deepmd.pt.model.model import (
    DipoleModel,
    DOSModel,
    DPZBLModel,
    EnergyModel,
    LinearEnergyModel,
    PolarModel,
    PropertyModel,
    SpinEnergyModel,
)
from deepmd.pt.model.task import (
    DipoleFittingNet,
    DOSFittingNet,
    EnergyFittingNet,
    PolarFittingNet,
    PropertyFittingNet,
)
from deepmd.utils.spin import (
    Spin,
)

from ....consistent.common import (
    parameterized,
)
from ...common.cases.model.model import (
    DipoleModelTest,
    DosModelTest,
    EnerModelTest,
    LinearEnerModelTest,
    PolarModelTest,
    PropertyModelTest,
    SpinEnerModelTest,
    ZBLModelTest,
)
from ...dpmodel.descriptor.test_descriptor import (
    DescriptorParamDPA1,
    DescriptorParamDPA1List,
    DescriptorParamDPA2,
    DescriptorParamDPA2List,
    DescriptorParamDPA3,
    DescriptorParamDPA3List,
    DescriptorParamHybrid,
    DescriptorParamHybridMixed,
    DescriptorParamHybridMixedTTebd,
    DescriptorParamSeA,
    DescriptorParamSeAList,
    DescriptorParamSeR,
    DescriptorParamSeRList,
    DescriptorParamSeT,
    DescriptorParamSeTList,
    DescriptorParamSeTTebd,
    DescriptorParamSeTTebdList,
)
from ...dpmodel.fitting.test_fitting import (
    FittingParamDipole,
    FittingParamDipoleList,
    FittingParamDos,
    FittingParamDosList,
    FittingParamEnergy,
    FittingParamEnergyList,
    FittingParamPolar,
    FittingParamPolarList,
    FittingParamProperty,
    FittingParamPropertyList,
)
from ...dpmodel.model.test_model import (
    skip_model_tests,
)
from ..backend import (
    PTTestCase,
)

defalut_des_param = [
    DescriptorParamSeA,
    DescriptorParamSeR,
    DescriptorParamSeT,
    DescriptorParamSeTTebd,
    DescriptorParamDPA1,
    DescriptorParamDPA2,
    DescriptorParamDPA3,
    DescriptorParamHybrid,
    DescriptorParamHybridMixed,
]
defalut_fit_param = [
    FittingParamEnergy,
    FittingParamDos,
    FittingParamDipole,
    FittingParamPolar,
    FittingParamProperty,
]


@parameterized(
    des_parameterized=(
        (
            *[(param_func, DescrptDPA3) for param_func in DescriptorParamDPA3List],
        ),  # descrpt_class_param & class
        ((FittingParamEnergy, EnergyFittingNet),),  # fitting_class_param & class
    ),
    fit_parameterized=(
        (
            (DescriptorParamDPA3, DescrptDPA3),
        ),  # descrpt_class_param & class
        (
            *[(param_func, EnergyFittingNet) for param_func in FittingParamEnergyList],
        ),  # fitting_class_param & class
    ),
)
class TestEnergyModelPT(unittest.TestCase, EnerModelTest, PTTestCase):
    @property
    def modules_to_test(self):
        skip_test_jit = getattr(self, "skip_test_jit", False)
        modules = PTTestCase.modules_to_test.fget(self)
        if not skip_test_jit:
            # for Model, we can test script module API
            modules += [
                self._script_module
                if hasattr(self, "_script_module")
                else self.script_module
            ]
        return modules

    @classmethod
    def setUpClass(cls) -> None:
        EnerModelTest.setUpClass()
        (DescriptorParam, Descrpt) = cls.param[0]
        (FittingParam, Fitting) = cls.param[1]
        # set special precision
        if Descrpt in [DescrptDPA2]:
            cls.epsilon_dict["test_smooth"] = 1e-8
        if Descrpt in [DescrptSeT, DescrptSeTTebd]:
            # computational expensive
            cls.expected_sel = [i // 4 for i in cls.expected_sel]
            cls.expected_rcut = cls.expected_rcut / 2
        cls.input_dict_ds = DescriptorParam(
            len(cls.expected_type_map),
            cls.expected_rcut,
            cls.expected_rcut / 2,
            cls.expected_sel,
            cls.expected_type_map,
        )

        # set skip tests
        skiptest, skip_reason = skip_model_tests(cls)
        if skiptest:
            raise cls.skipTest(cls, skip_reason)

        ds = Descrpt(**cls.input_dict_ds)
        cls.input_dict_ft = FittingParam(
            ntypes=len(cls.expected_type_map),
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
            type_map=cls.expected_type_map,
        )
        ft = Fitting(
            **cls.input_dict_ft,
        )
        cls.module = EnergyModel(
            ds,
            ft,
            type_map=cls.expected_type_map,
        )
        # only test jit API once for different models
        # if (
        #     DescriptorParam not in defalut_des_param
        #     or FittingParam not in defalut_fit_param
        # ):
        cls.skip_test_jit = True
        # else:
        #     with torch.jit.optimized_execution(False):
        #         cls._script_module = torch.jit.script(cls.module)
        cls.output_def = cls.module.translated_output_def()
        cls.expected_has_message_passing = ds.has_message_passing()
        cls.expected_sel_type = ft.get_sel_type()
        cls.expected_dim_fparam = ft.get_dim_fparam()
        cls.expected_dim_aparam = ft.get_dim_aparam()


if __name__ == "__main__":
    unittest.main()
