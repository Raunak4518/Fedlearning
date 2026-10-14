//@@ HANDOUT
C.push(H2("5.9 Proof on real data: what each component does"));
C.push(P("Measured on real features: MNIST and FashionMNIST, long tail, 3 seeds, the same splits as every run. The *referee* is a head trained on held-out real features; it never takes part in training. Code: experiments/D01_component_evidence.py."));
C.push(...img("ev01_collapse.png", 640, "Rows of rare classes under FedAvg end at 0.33 (MNIST) and 0.43 (FMNIST) of their start value; with HWA they grow (×1.7, ×2.4). They shrink even when 70–90 % of clients hold the class."));
C.push(...img("ev02_fidelity.png", 640, "Rare-class fake samples recognised as their class: GeFL-F 11 % / 21 %, + HWA 33 % / 67 %, PC-VAE 93 % / 90 %, Ours-A 89 % / 80 % (real test features: 96 % / 84 %)."));
C.push(...img("ev03_moments.png", 640, "Spread of fake features (1 = real): PC-VAE 0.50 / 0.45, + MC 0.86 / 0.89, + KH 0.95 / 0.98. Rare-class mean error: 0.19 → 0.09 → 0.05 class std (MNIST)."));
C.push(...img("ev11_fake_separability.png", 600, "Class separability F of fake features: GeFL-F 0.10 / 0.28 (classes overlap), PC-VAE 0.78 / 1.36 (too clean), Ours-A 0.35 / 0.62, real 0.33 / 0.61."));
C.push(...img("ev04_mmd.png", 640, "Distance to real test features (MMD, all classes): GeFL-F 0.22 / 0.25 → PC-VAE 0.18 / 0.19 → + MC 0.086 / 0.092 → + KH 0.069 / 0.074. Lower MMD goes with higher real accuracy (r = −0.75 / −0.51), as Lemma 1 says."));
C.push(...img("ev05_tstr.png", 640, "Head trained only on fake features, tested on real data (balanced accuracy): GeFL-F 59 / 47, + HWA 81 / 74, PC-VAE 92 / 77, + MC 94 / 78, + KH 94 / 78. Rare classes on MNIST: 32 → 91. KH adds little in this test; its gain shows inside the full pipeline (+0.8 / +0.7)."));
C.push(...img("ev07_rare_class_mnist.png", 640, "MNIST's rarest class (24 real samples): GeFL-F's fake class 9 lies on other classes; PC-VAE is on the class but too narrow; MC and KH spread it over the real class."));
C.push(...img("ev08_la_csl.png", 640, "LA (seed 0): balanced accuracy 92.3 → 95.5 (MNIST), 74.0 → 79.2 (FMNIST); correlation between a client's own class shares and its predictions 0.61 → 0.42, 0.67 → 0.54. CSL: the soft target is closer to the real-data referee than the one-hot label (TV 0.098 → 0.088, 0.192 → 0.179)."));
C.push(...img("ev09_ts.png", 520, "More fake training helps more with the better generator (MNIST: −0.05 vs +1.71)."));
//@@ HANDOUT_SEP
C.push(...img("ev06_separability.png", 640, "Class means are far apart on MNIST, FMNIST and CIFAR-10, but almost identical on SVHN (nearest-mean accuracy near chance): the reason Ours-A is weak on SVHN."));
//@@ PCVAE
C.push(H2("Proof: real data pe (3 seeds)"));
C.push(...img("ev01_collapse.png", 600, "Har line = ek class ki row ka size (start = 1). GeFL-F: rare classes ki rows 0.33 (MNIST) / 0.43 (FMNIST) tak gir jaati hain. HWA: sab badhti hain."));
C.push(P("**Plot kaise padhein:** left panel mein light lines (rare classes) neeche gir rahi hain, dark lines (common) upar. Right panel: har point ek class hai; 1 se neeche = row shrink hui. Dhyan do: 70–90 % clients ke paas class hone pe bhi row giri, kyunki holders ka data gradient chhota hai aur Adam weight decay ko poora step bana deta hai."));
C.push(...img("ev02_fidelity.png", 600, "Referee (real data pe trained head) fake samples ko kitna sahi pehchanta hai."));
C.push(P("**Matlab:** GeFL-F ke rare-class fake samples sirf 11 % (MNIST) / 21 % (FMNIST) baar sahi pehchane jaate hain, PC-VAE ke 93 % / 90 %. Real test features 96 % / 84 % pe hain, to PC-VAE real ke barabar class information deta hai."));
C.push(...img("ev07_rare_class_mnist.png", 600, "MNIST ki rarest class (sirf 24 real samples): green = real, colour = fake."));
C.push(P("**Matlab:** GeFL-F ka fake class 9 doosri classes ke upar pada hai (grey blob). PC-VAE class ke upar hai lekin bahut sikuda hua (too narrow). MC aur KH usse real class jitna phaila dete hain."));
C.push(...img("ev06_separability.png", 600, "Class mean kitna class pehchanta hai: F aur nearest-mean accuracy."));
C.push(P("**Matlab:** MNIST, FMNIST, CIFAR-10 pe class means door-door hain, to mean se class pehchani jaati hai. SVHN pe sab means ek jagah hain (accuracy chance ke paas), isliye PC-VAE wahan kaam nahi karta."));
//@@ MC
C.push(H2("Proof: real data pe (3 seeds)"));
C.push(...img("ev03_moments.png", 600, "Left: spread (1 = real). Right: class mean ki galti (0 = exact)."));
C.push(P("**Matlab:** PC-VAE ka spread sirf 0.50 / 0.45 hai (samples sikude hue). MC ke baad 0.86 / 0.89, KH ke baad 0.95 / 0.98. Exact 1.0 nahi aata kyunki map ke baad ReLU negative values kaat deta hai, aur ye honest baat hai. Rare class ke mean ki galti 0.19 → 0.09 class std (MNIST)."));
C.push(...img("ev11_fake_separability.png", 600, "Fake features ki class separability F; green band = real features."));
C.push(P("**Matlab:** GeFL-F ki fake classes ek doosre mein ghusi hain (F bahut kam). PC-VAE ki fake classes zaroorat se zyada alag hain (too clean, F 0.78 vs real 0.33). Head too-clean data pe aasaan cheez seekhta hai jo real pe kaam nahi aati. MC + KH ke baad F 0.35 / 0.62, real 0.33 / 0.61 ke almost barabar."));
//@@ KH
C.push(H2("Proof: real data pe (3 seeds)"));
C.push(...img("ev04_mmd.png", 600, "Left: har class ka MMD (fake ↔ real test). Right: MMD vs real accuracy, har point = ek (class, method)."));
C.push(P("**Matlab:** MMD (sab classes): GeFL-F 0.22 → PC-VAE 0.18 → MC 0.086 → KH 0.069 (MNIST). Right plot Lemma 1 ko real data pe dikhata hai: jahan MMD kam, wahan fake pe trained head real data pe zyada sahi (r = −0.75 MNIST, −0.51 FMNIST)."));
C.push(...img("ev05_tstr.png", 600, "Head sirf fake features pe train, real test pe check."));
C.push(P("**Matlab:** sirf fake data se seekh ke real accuracy: GeFL-F 59 %, Ours-A 94 % (MNIST); real data pe trained head 97 %. Rare classes 32 % → 91 %. Honest baat: is test mein KH MC ke upar thoda hi jodta hai (+0.5 MNIST, +0.2 FMNIST); poori pipeline mein uska fayda +0.8 / +0.7 hai."));
//@@ CSL
C.push(H2("Proof: real data pe (seed 0)"));
C.push(...img("ev08_la_csl.png", 600, "Right column = CSL: training target ki real-data referee se doori (TV, kam = behtar)."));
C.push(P("**Matlab:** right column dekho. Grey = one-hot label, blue = consensus soft label. Lagbhag har class mein blue chhota hai: TV 0.098 → 0.088 (MNIST), 0.192 → 0.179 (FMNIST). Matlab CSL ka target real data ki sachchai ke zyada paas hai. Sabse bada fark confusing classes pe (MNIST class 8, FMNIST class 6 = shirt)."));
//@@ LA
C.push(H2("Proof: real data pe (seed 0)"));
C.push(P("Upar waali figure (Section 5) ke left aur middle column LA ke liye hain:"));
C.push(B("**Left:** test set balanced hai, to har class ko 10 % predictions milni chahiye (green line). Plain CE common classes ko zyada predict karta hai (FMNIST class 0: 16.6 %), rare ko kam (class 6: 3.8 %). LA ke saath bars 10 % ke paas: 13.3 % aur 5.6 %."));
C.push(B("**Middle:** har point = ek client ki ek class. x = client ke data mein us class ka share, y = head us class ko kitna predict karta hai. CE mein correlation 0.61 / 0.67 (head apni skew copy karta hai), LA mein 0.42 / 0.54."));
C.push(B("**Accuracy:** CE 92.3 → LA 95.5 (MNIST), 74.0 → 79.2 (FMNIST), rare classes 84.8 → 92.6 (MNIST)."));
//@@ TS
C.push(H2("Proof: plot"));
C.push(...img("ev09_ts.png", 520, "T_s = 1 se 10: grey = purana generator, blue = Ours-A generator."));
C.push(P("**Matlab:** MNIST pe purane generator ke saath line flat hai (−0.05), Ours-A generator ke saath upar jaati hai (+1.71). Yahi Proposition 5 kehta hai."));
