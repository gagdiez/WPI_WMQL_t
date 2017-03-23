import sys
from pipelib import Src, GeneratedNode, needDeps
from pipelines.pnlscripts.util import TemporaryDirectory
import pipelib
import software
import plumbum
from plumbum import local, FG

DEFAULT_TARGET = 'dwimask'


class DwiMaskSlicer(GeneratedNode):
    def __init__(self, caseid, dwi, version_Slicer, hash_mrtrix3):
        self.deps = [dwi]
        self.params = [version_Slicer, hash_mrtrix3]
        self.ext = '.nii.gz'
        GeneratedNode.__init__(self, locals())

    def build(self):
        needDeps(self)
        with TemporaryDirectory() as tmpdir, local.cwd(
                tmpdir), software.mrtrix3.env(self.hash_mrtrix3):
            from plumbum.cmd import maskfilter
            Slicer = local[software.Slicer.getPath(self.version_Slicer)]
            Slicer['--launch', 'DiffusionWeightedVolumeMasking', self.dwi.path(
            ), 'b0.nrrd', 'otsumask.nrrd', '--baselineBValueThreshold', '1000',
                   '--removeislands'] & FG
            Slicer['--launch', 'ResampleScalarVolume', 'otsumask.nrrd',
                   'otsumask.nii'] & FG
            maskfilter['-scale', 2, self.path(), 'clean', 'otsumask.nii',
                       '-force'] & FG

class DiceCoefficient(GeneratedNode):
    def __init__(self, caseid, maskManual, mask, hash_BRAINSTools):
        self.deps = [maskManual, mask]
        self.params = [version_BRAINSTools]
        self.ext = '.txt'
        GeneratedNode.__init__(self, locals())

    def build(self):
        from plumbum.cmd import ImageMath
        needDeps(self)
        with TemporaryDirectory() as tmpdir, software.BRAINSTools(self.version_BRAINSTools):
            tmptxt = tmpdir / 'dice.txt'
            ImageMath[3, tmptxt, "DiceAndMinDistSum", self.maskManual, self.mask]
            with open(tmptxt, 'r') as f:
                coeff = f.read().split('')[-1]
            with open(self.path(), 'w') as f:
                f.write(coeff)


def makePipeline(caseid,
                 dwiPathKey='dwi',
                 dwimaskManualPathKey='dwimaskManual',
                 version_Slicer='4.7.0',
                 hash_BRAINSTools='41353e8',
                 hash_mrtrix3='97e4b3b'):
    pipeline = {'_name': "dwi masking test"}
    pipeline['dwi'] = Src(caseid, dwiPathKey)
    pipeline['dwimaskManual'] = Src(caseid, dwimaskManualPathKey)
    pipeline['dwimask'] = DwiMaskSlicer(caseid, pipeline['dwi'],
                                        version_Slicer, hash_mrtrix3)
    pipeline['dice'] = DiceCoefficient(caseid, pipeline['dwimaskManual'], pipeline['dwimask'], hash_BRAINSTools)
    return pipeline

# /rfanfs/pnl-zorro/projects/Lyall_R03/Slicer-build2/Slicer-build/Slicer --launch DiffusionWeightedVolumeMasking  dwi.nhdr dwi_b0.nrrd dwi_OTSUtensormask.nrrd --baselineBValueThreshold 1000 --removeislands
# /rfanfs/pnl-zorro/projects/Lyall_R03/Slicer-build2/Slicer-build/Slicer --launch  ResampleScalarVolume dwi_OTSUtensormask.nrrd dwi_OTSUtensormask.nii
# maskfilter -scale 2 dwi_OTSUtensormask_cleaned.nii clean  dwi_OTSUtensormask.nii -force
