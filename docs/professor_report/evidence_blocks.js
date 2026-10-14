//@@ HANDOUT
C.push(H2("5.9 Proof on real data: what each component does"));
C.push(P("Measured on real features: MNIST and FashionMNIST, long tail, 3 seeds, the same splits as every run (Kaggle T4; a second run on another GPU gives the same numbers within about 1 point). The *referee* is a head trained on held-out real features; it never takes part in training. Code: experiments/D01_component_evidence.py."));
C.push(...img("ev01_collapse.png", 640, "Rows of rare classes under FedAvg end at 0.33 (MNIST) and 0.44 (FMNIST) of their start value; with HWA they grow (×1.7, ×2.4). They shrink even when 70–90 % of clients hold the class."));
C.push(...img("ev02_fidelity.png", 640, "Rare-class fake samples recognised as their class: GeFL-F 10 % / 20 %, + HWA 31 % / 71 %, PC-VAE 93 % / 88 %, Ours-A 90 % / 79 % (real test features: 97 % / 84 %)."));
C.push(...img("ev03_moments.png", 640, "Spread of fake features (1 = real): PC-VAE 0.47 / 0.45, + MC 0.85 / 0.90, + KH 0.95 / 0.98. Rare-class mean error: 0.22 → 0.09 → 0.05 class std (MNIST)."));
C.push(...img("ev11_fake_separability.png", 600, "Class separability F of fake features: GeFL-F 0.09 / 0.32 (classes overlap), PC-VAE 0.78 / 1.35 (too clean), Ours-A 0.32 / 0.61, real 0.30 / 0.60."));
C.push(...img("ev04_mmd.png", 640, "Distance to real test features (MMD, all classes): GeFL-F 0.22 / 0.25 → PC-VAE 0.20 / 0.19 → + MC 0.089 / 0.093 → + KH 0.071 / 0.078. Lower MMD goes with higher real accuracy (r = −0.69 / −0.47), as Lemma 1 says."));
C.push(...img("ev05_tstr.png", 640, "Head trained only on fake features, tested on real data (balanced accuracy): GeFL-F 57 / 48, + HWA 81 / 76, PC-VAE 92 / 76, + MC 94 / 77, + KH 94 / 76. Rare classes on MNIST: 32 → 89. KH does not add in this test (−0.2 / −1.1); its gain shows only inside the full pipeline (+0.8 / +0.7)."));
C.push(...img("ev07_rare_class_mnist.png", 640, "MNIST's rarest class (24 real samples): GeFL-F's fake class 9 lies on other classes; PC-VAE is on the class but too narrow; MC and KH spread it over the real class."));
C.push(...img("ev08_la_csl.png", 640, "LA (seed 0): balanced accuracy 92.0 → 95.2 (MNIST), 75.5 → 79.9 (FMNIST); correlation between a client's own class shares and its predictions 0.64 → 0.44, 0.67 → 0.52. CSL: the soft target is closer to the real-data referee than the one-hot label, lower or equal in every class (TV 0.100 → 0.092, 0.212 → 0.202)."));
C.push(...img("ev09_ts.png", 520, "More fake training helps more with the better generator (MNIST: −0.05 vs +1.71)."));
C.push(...img("ev06_separability.png", 640, "How well the exact class mean identifies the class (3 seeds): F = 0.30 / 0.60 / 0.07 / 0.012 and nearest-mean accuracy 82 / 69 / 41 / 12 % on MNIST / FMNIST / CIFAR-10 / SVHN (chance 10 %). SVHN's means carry almost no class information: the reason Ours-A is weak there."));
//@@ PCVAE
C.push(H2("Proof: real data pe (3 seeds, Kaggle)"));
C.push(...img("ev01_collapse.png", 600, "Har line = ek class ki row ka size (start = 1). GeFL-F: rare classes ki rows 0.33 (MNIST) / 0.44 (FMNIST) tak gir jaati hain. HWA: sab badhti hain."));
C.push(P("**Plot kaise padhein:** left panel mein light lines (rare classes) neeche gir rahi hain, dark lines (common) upar. Right panel: har point ek class hai; 1 se neeche = row shrink hui. Dhyan do: 70–90 % clients ke paas class hone pe bhi row giri, kyunki holders ka data gradient chhota hai aur Adam weight decay ko poora step bana deta hai."));
C.push(...img("ev02_fidelity.png", 600, "Referee (real data pe trained head) fake samples ko kitna sahi pehchanta hai."));
C.push(P("**Matlab:** GeFL-F ke rare-class fake samples sirf 10 % (MNIST) / 20 % (FMNIST) baar sahi pehchane jaate hain, PC-VAE ke 93 % / 88 %. Real test features 97 % / 84 % pe hain, to PC-VAE real ke barabar class information deta hai."));
C.push(...img("ev07_rare_class_mnist.png", 600, "MNIST ki rarest class (sirf 24 real samples, 8/10 clients ke paas): green = real, colour = fake."));
C.push(P("**Matlab:** GeFL-F ka fake class 9 doosri classes ke upar pada hai (grey blob). PC-VAE class ke upar hai lekin bahut sikuda hua (too narrow). MC aur KH usse real class jitna phaila dete hain."));
C.push(...img("ev06_separability.png", 600, "Class mean kitna class pehchanta hai: F aur nearest-mean accuracy."));
C.push(P("**Matlab:** sirf class mean se class pehchanne ki accuracy: MNIST 82 %, FMNIST 69 %, CIFAR-10 41 %, SVHN 12 % (chance 10 %). Figure ke neeche waale 4 panels mein X = class means: MNIST/FMNIST pe door-door, CIFAR-10 pe thode paas, SVHN pe sab ek jagah. Isliye PC-VAE MNIST/FMNIST pe sabse achha, CIFAR pe theek, SVHN pe kamzor."));
//@@ MC
C.push(H2("Proof: real data pe (3 seeds, Kaggle)"));
C.push(...img("ev03_moments.png", 600, "Left: spread (1 = real). Right: class mean ki galti (0 = exact)."));
C.push(P("**Matlab:** PC-VAE ka spread sirf 0.47 / 0.45 hai (samples sikude hue). MC ke baad 0.85 / 0.90, KH ke baad 0.95 / 0.98. Exact 1.0 nahi aata kyunki map ke baad ReLU negative values kaat deta hai, aur ye honest baat hai. Rare class ke mean ki galti 0.22 → 0.09 class std (MNIST)."));
C.push(...img("ev11_fake_separability.png", 600, "Fake features ki class separability F; green band = real features."));
C.push(P("**Matlab:** GeFL-F ki fake classes ek doosre mein ghusi hain (F bahut kam). PC-VAE ki fake classes zaroorat se zyada alag hain (too clean, F 0.78 vs real 0.30). Head too-clean data pe aasaan cheez seekhta hai jo real pe kaam nahi aati. MC + KH ke baad F 0.32 / 0.61, real 0.30 / 0.60 ke almost barabar."));
//@@ KH
C.push(H2("Proof: real data pe (3 seeds, Kaggle)"));
C.push(...img("ev04_mmd.png", 600, "Left: har class ka MMD (fake ↔ real test). Right: MMD vs real accuracy, har point = ek (class, method)."));
C.push(P("**Matlab:** MMD (sab classes): GeFL-F 0.22 → PC-VAE 0.20 → MC 0.089 → KH 0.071 (MNIST). Right plot Lemma 1 ko real data pe dikhata hai: jahan MMD kam, wahan fake pe trained head real data pe zyada sahi (r = −0.69 MNIST, −0.47 FMNIST)."));
C.push(...img("ev05_tstr.png", 600, "Head sirf fake features pe train, real test pe check."));
C.push(P("**Matlab:** sirf fake data se seekh ke real accuracy: GeFL-F 57 %, Ours-A 94 % (MNIST); real data pe trained head 97 %. Rare classes 32 % → 89 %. Honest baat: is test mein KH MC ke upar kuch nahi jodta (−0.2 MNIST, −1.1 FMNIST); uska fayda sirf poori pipeline mein dikhta hai (+0.8 / +0.7). Professor pooche to yahi bolna: KH MMD ghataata hai (upar wala plot), lekin akele head ke test mein MC hi kaafi hai."));
//@@ CSL
C.push(H2("Proof: real data pe (seed 0, Kaggle)"));
C.push(...img("ev08_la_csl.png", 600, "Right column = CSL: training target ki real-data referee se doori (TV, kam = behtar)."));
C.push(P("**Matlab:** right column dekho. Grey = one-hot label, blue = consensus soft label. **Har class** mein blue chhota ya barabar hai: TV 0.100 → 0.092 (MNIST), 0.212 → 0.202 (FMNIST). Matlab CSL ka target real data ki sachchai ke zyada paas hai. Sabse bada fark confusing classes pe (MNIST class 8, FMNIST class 6 = shirt)."));
//@@ LA
C.push(H2("Proof: real data pe (seed 0, Kaggle)"));
C.push(P("Upar waali figure (Section 5) ke left aur middle column LA ke liye hain:"));
C.push(B("**Left:** test set balanced hai, to har class ko 10 % predictions milni chahiye (green line). Plain CE common classes ko zyada predict karta hai (FMNIST class 0: 16.4 %), rare ko kam (class 6: 3.7 %). LA ke saath bars 10 % ke paas: 13.1 % aur 5.7 %."));
C.push(B("**Middle:** har point = ek client ki ek class. x = client ke data mein us class ka share, y = head us class ko kitna predict karta hai. CE mein correlation 0.64 / 0.67 (head apni skew copy karta hai), LA mein 0.44 / 0.52."));
C.push(B("**Accuracy:** CE 92.0 → LA 95.2 (MNIST), 75.5 → 79.9 (FMNIST), rare classes 84.5 → 91.9 (MNIST)."));
//@@ TS
C.push(H2("Proof: plot"));
C.push(...img("ev09_ts.png", 520, "T_s = 1 se 10: grey = purana generator, blue = Ours-A generator."));
C.push(P("**Matlab:** MNIST pe purane generator ke saath line flat hai (−0.05), Ours-A generator ke saath upar jaati hai (+1.71). Yahi Proposition 5 kehta hai."));
