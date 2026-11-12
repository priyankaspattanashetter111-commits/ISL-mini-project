# Dataset decision and verified limits

Reviewed on 12 November 2026. The chosen primary source is new webcam collection because the student selected that route. No external dataset was downloaded for training and no external dataset's folder labels were locally verified.

## Primary local dataset

Obtain samples with `vision.collect` after human review and signer agreement. Folder layout is `data/raw/<signer>/<session>/<letter>/<clip>.npz`. A-Z in the registry are intended targets; actual supported classes are only those reviewed and collected. Files contain timestamped landmarks, not pictures. Rights come from the participating people and your institution's project agreement; no public reuse licence is assumed. Document permission separately without putting real participant identities in the repository.

## ISLAN candidate

Source: https://data.mendeley.com/datasets/rc349j45m5/1
DOI: 10.17632/rc349j45m5.1. Contributors: Elakkiya R and Rajalakshmi E. Licence: CC BY 4.0; give attribution, link the licence and identify changes if reused.

The publisher describes alphabet/number JPEG images plus videos, separating signers and one-/two-hand variants, and mentions motion videos for J and Z. Use Download All on the dataset page. Its public file API returned HTTP 403 during this work, so archive names, extracted folders and actual labels were not inspected. Do not claim a verified 26-class mapping. If you obtain the archive later, inventory its actual files, variants and signer IDs before importing it. It is a candidate for isolated-letter experiments, not proof of naturally continuous fingerspelling.

## INCLUDE comparison

Author source: https://github.com/AI4Bharat/sign-language.ai4bharat.org/blob/master/docs/INCLUDE.md
Dataset card: https://huggingface.co/datasets/ai4bharat/INCLUDE
Downloads: https://zenodo.org/records/4010759

INCLUDE is an isolated Indian Sign Language vocabulary-sign resource. The author card labels it CC BY 4.0 and provides video download instructions. The Hugging Face metadata uses parent_label, label and video_path with train/val/test entries; videos are downloaded separately. The actual class inventory has not been inspected here, and no alphabet mapping is assumed. It is not selected for A-Z training or continuous-letter segmentation. Attribute Sridhar, Ganesan, Kumar and Khapra and cite the INCLUDE publication if it is later used.

## Continuous language datasets

iSign (https://exploration-lab.github.io/iSign/) and ISLTranslate (https://arxiv.org/abs/2307.05440) are research leads for sentence/phrase-level ISL. Their letters, licences, file layouts and aligned fingerspelling boundaries have not been verified in this project, so neither is selected or imported. Sentence translation data cannot automatically supply A-Z boundary supervision.

## Reference material

ISLRTC dictionary portal: https://islrtc.nic.in/isl-dictionary/
Use references to arrange a human review. Do not assume a public portal grants redistribution rights for every video, or that ASL and Indian Sign Language use the same alphabet gestures. Do not mix local one- and two-hand variants without recording the variant and evaluating them separately.
