"""Reference database for the report, formatted in APA 7th edition style.
Each entry: authors (list of "Surname, I. I."), year, and a pre-formatted APA tail (title onward) using *italic* markup."""

REFS = {
    "afchar2018mesonet": (["Afchar, D.", "Nozick, V.", "Yamagishi, J.", "Echizen, I."], "2018",
        "MesoNet: A compact facial video forgery detection network. In *2018 IEEE International Workshop on Information Forensics and Security (WIFS)* (pp. 1–7). IEEE."),
    "ba2016layernorm": (["Ba, J. L.", "Kiros, J. R.", "Hinton, G. E."], "2016",
        "*Layer normalization* (arXiv:1607.06450). arXiv. https://doi.org/10.48550/arXiv.1607.06450"),
    "cheng2025surlid": (["Cheng, J.", "Yan, Z.", "Zhang, Y.", "Hao, L.", "Ai, J.", "Zou, Q.", "Li, C.", "Wang, Z."], "2025",
        "Stacking brick by brick: Aligned feature isolation for incremental face forgery detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*."),
    "chollet2017xception": (["Chollet, F."], "2017",
        "Xception: Deep learning with depthwise separable convolutions. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 1251–1258)."),
    "cui2025forensicsadapter": (["Cui, X.", "Li, Y.", "Luo, A.", "Zhou, J.", "Dong, J."], "2025",
        "Forensics Adapter: Adapting CLIP for generalizable face forgery detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 19207–19217)."),
    "delange2022survey": (["De Lange, M.", "Aljundi, R.", "Masana, M.", "Parisot, S.", "Jia, X.", "Leonardis, A.", "Slabaugh, G.", "Tuytelaars, T."], "2022",
        "A continual learning survey: Defying forgetting in classification tasks. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 44*(7), 3366–3385."),
    "deng2020retinaface": (["Deng, J.", "Guo, J.", "Ververas, E.", "Kotsia, I.", "Zafeiriou, S."], "2020",
        "RetinaFace: Single-shot multi-level face localisation in the wild. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 5203–5212)."),
    "dolhansky2019dfdcpreview": (["Dolhansky, B.", "Howes, R.", "Pflaum, B.", "Baram, N.", "Canton Ferrer, C."], "2019",
        "*The Deepfake Detection Challenge (DFDC) preview dataset* (arXiv:1910.08854). arXiv. https://doi.org/10.48550/arXiv.1910.08854"),
    "dosovitskiy2021vit": (["Dosovitskiy, A.", "Beyer, L.", "Kolesnikov, A.", "Weissenborn, D.", "Zhai, X.", "Unterthiner, T.", "Dehghani, M.", "Minderer, M.", "Heigold, G.", "Gelly, S.", "Uszkoreit, J.", "Houlsby, N."], "2021",
        "An image is worth 16x16 words: Transformers for image recognition at scale. In *International Conference on Learning Representations (ICLR)*."),
    "dufour2019dfd": (["Dufour, N.", "Gully, A."], "2019",
        "Contributing data to deepfake detection research. *Google Research Blog*. https://research.google/blog/contributing-data-to-deepfake-detection-research/"),
    "efron1994bootstrap": (["Efron, B.", "Tibshirani, R. J."], "1994",
        "*An introduction to the bootstrap*. Chapman and Hall/CRC."),
    "fontana2025chronological": (["Fontana, F.", "Diko, A.", "Lanzino, R.", "Marini, M. R.", "Kaddar, B.", "Foresti, G. L.", "Cinque, L."], "2025",
        "*Revisiting deepfake detection: Chronological continual learning and the limits of generalization* (arXiv:2509.07993). arXiv. https://doi.org/10.48550/arXiv.2509.07993"),
    "goodfellow2014gan": (["Goodfellow, I.", "Pouget-Abadie, J.", "Mirza, M.", "Xu, B.", "Warde-Farley, D.", "Ozair, S.", "Courville, A.", "Bengio, Y."], "2014",
        "Generative adversarial nets. In *Advances in Neural Information Processing Systems 27* (pp. 2672–2680)."),
    "guo2017calibration": (["Guo, C.", "Pleiss, G.", "Sun, Y.", "Weinberger, K. Q."], "2017",
        "On calibration of modern neural networks. In *Proceedings of the 34th International Conference on Machine Learning (ICML)* (pp. 1321–1330)."),
    "han2025fcg": (["Han, Y.-H.", "Huang, T.-M.", "Hua, K.-L.", "Chen, J.-C."], "2025",
        "Towards more general video-based deepfake detection through facial component guided adaptation for foundation model. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 22995–23005)."),
    "hinton2015distilling": (["Hinton, G.", "Vinyals, O.", "Dean, J."], "2015",
        "*Distilling the knowledge in a neural network* (arXiv:1503.02531). arXiv. https://doi.org/10.48550/arXiv.1503.02531"),
    "ho2020ddpm": (["Ho, J.", "Jain, A.", "Abbeel, P."], "2020",
        "Denoising diffusion probabilistic models. In *Advances in Neural Information Processing Systems 33* (pp. 6840–6851)."),
    "holm1979": (["Holm, S."], "1979",
        "A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70."),
    "kim2021cored": (["Kim, M.", "Tariq, S.", "Woo, S. S."], "2021",
        "CoReD: Generalizing fake media detection with continual representation using distillation. In *Proceedings of the 29th ACM International Conference on Multimedia (ACM MM)*."),
    "kim2026msfd": (["Kim, T.", "Choi, J.", "Jo, H.", "Park, B.", "Choi, J."], "2026",
        "*Preserving knowledge across space and time for continual video deepfake detection* (arXiv:2609.03446). arXiv. https://doi.org/10.48550/arXiv.2609.03446"),
    "kingma2015adam": (["Kingma, D. P.", "Ba, J."], "2015",
        "Adam: A method for stochastic optimization. In *International Conference on Learning Representations (ICLR)*."),
    "kirkpatrick2017ewc": (["Kirkpatrick, J.", "Pascanu, R.", "Rabinowitz, N.", "Veness, J.", "Desjardins, G.", "Rusu, A. A.", "Milan, K.", "Quan, J.", "Ramalho, T.", "Grabska-Barwinska, A.", "Hassabis, D.", "Clopath, C.", "Kumaran, D.", "Hadsell, R."], "2017",
        "Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences, 114*(13), 3521–3526."),
    "laiti2024prompt2guard": (["Laiti, F.", "Liberatori, B.", "De Min, T.", "Ricci, E."], "2024",
        "Conditioned prompt-optimization for continual deepfake detection. In *Proceedings of the International Conference on Pattern Recognition (ICPR)*."),
    "li2018lwf": (["Li, Z.", "Hoiem, D."], "2018",
        "Learning without forgetting. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 40*(12), 2935–2947."),
    "li2020celebdf": (["Li, Y.", "Yang, X.", "Sun, P.", "Qi, H.", "Lyu, S."], "2020",
        "Celeb-DF: A large-scale challenging dataset for DeepFake forensics. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 3207–3216)."),
    "li2020faceshifter": (["Li, L.", "Bao, J.", "Yang, H.", "Chen, D.", "Wen, F."], "2020",
        "Advancing high fidelity identity swapping for forgery detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 5074–5083)."),
    "li2020facexray": (["Li, L.", "Bao, J.", "Zhang, T.", "Yang, H.", "Chen, D.", "Wen, F.", "Guo, B."], "2020",
        "Face X-ray for more general face forgery detection. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 5001–5010)."),
    "ma2021dfwf": (["Ma, H.", "Yi, J.", "Tao, J.", "Bai, Y.", "Tian, Z.", "Wang, C."], "2021",
        "Continual learning for fake audio detection. In *Proceedings of Interspeech 2021* (pp. 886–890)."),
    "mccloskey1989catastrophic": (["McCloskey, M.", "Cohen, N. J."], "1989",
        "Catastrophic interference in connectionist networks: The sequential learning problem. In G. H. Bower (Ed.), *Psychology of learning and motivation* (Vol. 24, pp. 109–165). Academic Press."),
    "mirsky2021survey": (["Mirsky, Y.", "Lee, W."], "2021",
        "The creation and detection of deepfakes: A survey. *ACM Computing Surveys, 54*(1), Article 7."),
    "ojha2023univfd": (["Ojha, U.", "Li, Y.", "Lee, Y. J."], "2023",
        "Towards universal fake image detectors that generalize across generative models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 24480–24489)."),
    "pan2023dfil": (["Pan, K.", "Yin, Y.", "Wei, Y.", "Lin, F.", "Ba, Z.", "Liu, Z.", "Wang, Z.", "Cavallaro, L.", "Ren, K."], "2023",
        "DFIL: Deepfake incremental learning by exploiting domain-invariant forgery clues. In *Proceedings of the 31st ACM International Conference on Multimedia (ACM MM)*."),
    "paszke2019pytorch": (["Paszke, A.", "Gross, S.", "Massa, F.", "Lerer, A.", "Bradbury, J.", "Chanan, G.", "Killeen, T.", "Lin, Z.", "Gimelshein, N.", "Antiga, L.", "Desmaison, A.", "Köpf, A.", "Yang, E.", "DeVito, Z.", "Raison, M.", "Tejani, A.", "Chilamkurthy, S.", "Steiner, B.", "Fang, L.", "Bai, J.", "Chintala, S."], "2019",
        "PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32* (pp. 8024–8035)."),
    "platt1999probabilistic": (["Platt, J. C."], "1999",
        "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In A. J. Smola, P. Bartlett, B. Schölkopf, & D. Schuurmans (Eds.), *Advances in large margin classifiers* (pp. 61–74). MIT Press."),
    "radford2021clip": (["Radford, A.", "Kim, J. W.", "Hallacy, C.", "Ramesh, A.", "Goh, G.", "Agarwal, S.", "Sastry, G.", "Askell, A.", "Mishkin, P.", "Clark, J.", "Krueger, G.", "Sutskever, I."], "2021",
        "Learning transferable visual models from natural language supervision. In *Proceedings of the 38th International Conference on Machine Learning (ICML)* (pp. 8748–8763)."),
    "rebuffi2017icarl": (["Rebuffi, S.-A.", "Kolesnikov, A.", "Sperl, G.", "Lampert, C. H."], "2017",
        "iCaRL: Incremental classifier and representation learning. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2001–2010)."),
    "rombach2022ldm": (["Rombach, R.", "Blattmann, A.", "Lorenz, D.", "Esser, P.", "Ommer, B."], "2022",
        "High-resolution image synthesis with latent diffusion models. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 10684–10695)."),
    "rossler2019faceforensics": (["Rössler, A.", "Cozzolino, D.", "Verdoliva, L.", "Riess, C.", "Thies, J.", "Nießner, M."], "2019",
        "FaceForensics++: Learning to detect manipulated facial images. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)* (pp. 1–11)."),
    "shiohara2022sbi": (["Shiohara, K.", "Yamasaki, T."], "2022",
        "Detecting deepfakes with self-blended images. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 18720–18729)."),
    "song2023deepfakeface": (["Song, H.", "Huang, S.", "Dong, Y.", "Tu, W.-W."], "2023",
        "*Robustness and generalizability of deepfake detection: A study with diffusion models* (arXiv:2309.02218). arXiv. https://doi.org/10.48550/arXiv.2309.02218"),
    "sun2023hdp": (["Sun, K.", "Chen, S.", "Yao, T.", "Sun, X.", "Ding, S.", "Ji, R."], "2023",
        "*Continual face forgery detection via historical distribution preserving* (arXiv:2308.06217). arXiv. https://doi.org/10.48550/arXiv.2308.06217"),
    "thies2016face2face": (["Thies, J.", "Zollhöfer, M.", "Stamminger, M.", "Theobalt, C.", "Nießner, M."], "2016",
        "Face2Face: Real-time face capture and reenactment of RGB videos. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2387–2395)."),
    "thies2019neuraltextures": (["Thies, J.", "Zollhöfer, M.", "Nießner, M."], "2019",
        "Deferred neural rendering: Image synthesis using neural textures. *ACM Transactions on Graphics, 38*(4), Article 66."),
    "tolosana2020survey": (["Tolosana, R.", "Vera-Rodriguez, R.", "Fierrez, J.", "Morales, A.", "Ortega-Garcia, J."], "2020",
        "Deepfakes and beyond: A survey of face manipulation and fake detection. *Information Fusion, 64*, 131–148."),
    "vandermaaten2008tsne": (["van der Maaten, L.", "Hinton, G."], "2008",
        "Visualizing data using t-SNE. *Journal of Machine Learning Research, 9*(86), 2579–2605."),
    "wang2020alignuniform": (["Wang, T.", "Isola, P."], "2020",
        "Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In *Proceedings of the 37th International Conference on Machine Learning (ICML)* (pp. 9929–9939)."),
    "yan2023deepfakebench": (["Yan, Z.", "Zhang, Y.", "Yuan, X.", "Lyu, S.", "Wu, B."], "2023",
        "DeepfakeBench: A comprehensive benchmark of deepfake detection. In *Advances in Neural Information Processing Systems 36, Datasets and Benchmarks Track*."),
    "yan2025effort": (["Yan, Z.", "Wang, J.", "Jin, P.", "Zhang, K.-Y.", "Liu, C.", "Chen, S.", "Yao, T.", "Ding, S.", "Wu, B.", "Yuan, L."], "2025",
        "Orthogonal subspace decomposition for generalizable AI-generated image detection. In *Proceedings of the 42nd International Conference on Machine Learning (ICML)*."),
    "yermakov2026gend": (["Yermakov, A.", "Cech, J.", "Matas, J.", "Fritz, M."], "2026",
        "Deepfake detection that generalizes across benchmarks. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)* (pp. 773–783)."),
    "yu2020sdc": (["Yu, L.", "Twardowski, B.", "Liu, X.", "Herranz, L.", "Wang, K.", "Cheng, Y.", "Jui, S.", "van de Weijer, J."], "2020",
        "Semantic drift compensation for class-incremental learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 6982–6991)."),
    "zhang2023rawm": (["Zhang, X.", "Yi, J.", "Tao, J.", "Wang, C.", "Zhang, C. Y."], "2023",
        "Do you remember? Overcoming catastrophic forgetting for fake audio detection. In *Proceedings of the 40th International Conference on Machine Learning (ICML)* (pp. 41819–41831)."),
    "zhang2024rwm": (["Zhang, X.", "Yi, J.", "Wang, C.", "Zhang, C. Y.", "Zeng, S.", "Tao, J."], "2024",
        "What to remember: Self-adaptive continual learning for audio deepfake detection. In *Proceedings of the AAAI Conference on Artificial Intelligence, 38*(17), 19569–19577."),
    "zhang2025gpl": (["Zhang, X.", "Zhu, P.", "Zhang, C.", "Yan, Z.", "Cheng, J.", "Lao, M.", "Cai, S.", "Guo, Y."], "2025",
        "Generalization-preserved learning: Closing the backdoor to catastrophic forgetting in continual deepfake detection. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*."),
    "zi2020wilddeepfake": (["Zi, B.", "Chang, M.", "Chen, J.", "Ma, X.", "Jiang, Y.-G."], "2020",
        "WildDeepfake: A challenging real-world dataset for deepfake detection. In *Proceedings of the 28th ACM International Conference on Multimedia (ACM MM)* (pp. 2382–2390)."),
}


def surname(a):
    return a.split(",")[0]


def apa_authors(auth):
    if len(auth) == 1:
        return auth[0]
    if len(auth) <= 20:
        return ", ".join(auth[:-1]) + ", & " + auth[-1]
    return ", ".join(auth[:19]) + ", . . . " + auth[-1]


def apa_entry(key):
    auth, year, tail = REFS[key]
    a = apa_authors(auth)
    sep = " " if a.endswith(".") else ". "
    return f"{a}{sep}({year}). {tail}"


def intext(key):
    auth, year, _ = REFS[key]
    if len(auth) == 1:
        n = surname(auth[0])
    elif len(auth) == 2:
        n = f"{surname(auth[0])} & {surname(auth[1])}"
    else:
        n = f"{surname(auth[0])} et al."
    return n, year


def cite(keys):
    parts = []
    for k in keys:
        n, y = intext(k)
        parts.append(f"{n}, {y}")
    return "(" + "; ".join(parts) + ")"


def citet(key):
    n, y = intext(key)
    return f"{n.replace(' & ', ' and ')} ({y})"


def sort_key(key):
    auth, year, tail = REFS[key]
    s = auth[0].lower().replace("ö", "o").replace("van der ", "van der ")
    return (s, [x.lower() for x in auth[1:]], year)
