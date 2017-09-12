require(visNetwork)

# drawNetwork = function() {
  nodes = data.frame(id = 1:sum(sizes))
  edges = data.frame()
  
  nid = 1
  for (l in 1:num.layers) {
    for (n in 1:sizes[l]) {
      
      nodes$id[nid] = paste0("L",l,"N",n)
      
      if (l==1)
        nodes$label[nid] = paste0("I",n)
      else if (l==num.layers)
        nodes$label[nid] = paste0("O",n)
      else
        nodes$label[nid] = paste0("H",n)
      
      nodes$level[nid] = l
      nodes$shape="circle"
    
      nid = nid + 1
    }
  }
  
  eid = 1
  for (l in num.layers:2) {
    for (n in 1:sizes[l]) {
      for (nprev in 1:sizes[l-1]) {
        edges = rbind(edges, data.frame(
          from = paste0("L",l-1,"N",nprev),
          to = paste0("L",l,"N",n),
          label = round(weights[[l]][n,nprev],2),
          arrows = "to"
        ))
      }
    }
  }
  
  visNetwork(nodes, edges) %>% 
    visHierarchicalLayout(direction = "LR") 
# }

# drawNetwork()
