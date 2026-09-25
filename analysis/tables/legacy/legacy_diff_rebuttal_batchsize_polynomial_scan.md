| batch_size | Sven rank, legacy | Sven rank, fresh | change | best method, legacy | best method, fresh | Sven config, legacy | Sven config, fresh |
|---|---|---|---|---|---|---|---|
| 8 | 9/12 | 3/12 | -6 | SOAP | Muon | k=8, lr=0.5, rtol=0.01 | k=8, lr=0.5, rtol=1e-05 |
| 16 | 6/12 | 1/12 | -5 | SOAP | Sven | k=16, lr=0.5, rtol=0.01 | k=16, lr=0.5, rtol=0.01 |
| 32 | 7/12 | 1/12 | -6 | Polyak SGD | Sven | k=32, lr=0.1, rtol=0.0001 | k=32, lr=0.5, rtol=0.01 |
| 64 | 1/12 | 1/12 | = | Sven | Sven | k=64, lr=0.05, rtol=0.0001 | k=64, lr=0.5, rtol=0.01 |
| 128 | 2/12 | 5/12 | +3 | SOAP | MuonW | k=128, lr=0.05, rtol=0.001 | k=128, lr=0.1, rtol=0.001 |
| 256 | 1/12 | 4/12 | +3 | Sven | MuonW | k=256, lr=0.05, rtol=0.001 | k=256, lr=0.05, rtol=0.001 |
