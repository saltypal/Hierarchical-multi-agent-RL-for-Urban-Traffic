Simulation:
- Ticks:  
- 



Models used:
- Ward_Level : PPO, DQN, Q Learning
- Area_Level: GCN, STGCN
- City_Level: NetworkX
Reason why we are using a City Level as an Optimization algorithm is becaue a neural network would be an overkill and uneccesary as it only has to optimize and smoothen the traffic flow

---
# Area_level:

1. What are the inputs, What are the outputs:





STGCN:
graph TD
    Seq[Sequence of Node Features: X_t-k ... X_t] --> GCN[Spatial GCN Layer]
    GCN --> Spat[Spatial Representation Map]
    Spat --> GRU[Temporal GRU Cell]
    GRU --> Latent[Sequential Latent State]
    Latent --> Out[Linear + Sigmoid Output]
    Out --> Pred[Predicted Pressure per Ward]

