within ;
model SimpleRoom
  AixLib.ThermalZones.ReducedOrder.RC.OneElement
                thermalZoneOneElement(
    VAir=52.5,
    hConExt=2.7,
    hConWin=2.7,
    gWin=1,
    ratioWinConRad=0.09,
    nExt=1,
    RExt={0.00331421908725},
    CExt={5259932.23},
    hRad=5,
    RWin=0.01642857143,
    RExtRem=0.1265217391,
    nOrientations=2,
    AWin={7,7},
    ATransparent={7,7},
    AExt={3.5,8},
    redeclare package Medium = Modelica.Media.Air.SimpleAir,
    extWallRC(thermCapExt(each der_T(fixed=true))),
    energyDynamics=Modelica.Fluid.Types.Dynamics.FixedInitial,
    T_start=295.15) "Thermal zone"
    annotation (Placement(transformation(extent={{16,16},{64,52}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature preTem
    "Prescribed temperature for exterior walls outdoor surface temperature"
    annotation (Placement(transformation(extent={{-20,12},{-8,24}})));
  Modelica.Thermal.HeatTransfer.Components.Convection theConWall
    "Outdoor convective heat transfer of walls"
    annotation (Placement(transformation(extent={{8,24},{-2,14}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow perRad
    "Radiative heat flow of persons"
    annotation (Placement(transformation(extent={{20,-24},{40,-4}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedHeatFlow perCon
    "Convective heat flow of persons"
    annotation (Placement(transformation(extent={{20,-44},{40,-24}})));
  Modelica.Blocks.Sources.Constant hConWall(k=25*11.5)
    "Outdoor coefficient of heat transfer for walls"
    annotation (Placement(
    transformation(
    extent={{-4,-4},{4,4}},
    rotation=90,
    origin={2,2})));
  Modelica.Blocks.Sources.Constant const1[2](k=0)
    annotation (Placement(transformation(extent={{-22,60},{-2,80}})));
  Modelica.Blocks.Interfaces.RealInput Q_flow_heat
    annotation (Placement(transformation(extent={{-140,-44},{-100,-4}})));
  Modelica.Blocks.Math.Gain gain(k=0.5)
    annotation (Placement(transformation(extent={{-40,-30},{-28,-18}})));
  Modelica.Thermal.HeatTransfer.Components.Convection theConWall1
    "Outdoor convective heat transfer of walls"
    annotation (Placement(transformation(extent={{6,46},{-4,36}})));
  Modelica.Thermal.HeatTransfer.Sources.PrescribedTemperature preTem1
    "Prescribed temperature for exterior walls outdoor surface temperature"
    annotation (Placement(transformation(extent={{-20,34},{-8,46}})));
  Modelica.Blocks.Interfaces.RealOutput T_air
    annotation (Placement(transformation(extent={{100,30},{140,70}})));
  Modelica.Blocks.Interfaces.RealInput T_oda
    annotation (Placement(transformation(extent={{-140,20},{-100,60}})));
equation
  connect(perRad.port,thermalZoneOneElement. intGainsRad)
    annotation (Line(
    points={{40,-14},{72,-14},{72,42},{64,42}},
    color={191,0,0}));
  connect(thermalZoneOneElement.extWall,theConWall. solid)
    annotation (Line(points={{16,30},{12,30},{12,19},{8,19}},
    color={191,0,0}));
  connect(theConWall.fluid,preTem. port)
    annotation (Line(points={{-2,19},{-4,19},{-4,18},{-8,18}},
                                                           color={191,0,0}));
  connect(hConWall.y,theConWall. Gc)
    annotation (Line(points={{2,6.4},{2,14},{3,14}},      color={0,0,127}));
  connect(perCon.port,thermalZoneOneElement. intGainsConv)
    annotation (
    Line(points={{40,-34},{68,-34},{68,38},{64,38}}, color={191,0,0}));
  connect(const1.y, thermalZoneOneElement.solRad)
    annotation (Line(points={{-1,70},{8,70},{8,49},{15,49}}, color={0,0,127}));
  connect(Q_flow_heat, gain.u)
    annotation (Line(points={{-120,-24},{-41.2,-24}}, color={0,0,127}));
  connect(thermalZoneOneElement.window, theConWall1.solid) annotation (Line(
        points={{16,38},{12,38},{12,41},{6,41}}, color={191,0,0}));
  connect(hConWall.y, theConWall1.Gc)
    annotation (Line(points={{2,6.4},{2,22},{2,36},{1,36}}, color={0,0,127}));
  connect(theConWall1.fluid, preTem1.port) annotation (Line(points={{-4,41},{-6,
          41},{-6,40},{-8,40}}, color={191,0,0}));
  connect(gain.y, perCon.Q_flow)
    annotation (Line(points={{-27.4,-24},{14,-24},{14,-34},{20,-34}},
                                                           color={0,0,127}));
  connect(gain.y, perRad.Q_flow) annotation (Line(points={{-27.4,-24},{14,-24},
          {14,-14},{20,-14}}, color={0,0,127}));
  connect(thermalZoneOneElement.TAir, T_air) annotation (Line(points={{65,50},{
          120,50}},                 color={0,0,127}));
  connect(T_oda, preTem1.T) annotation (Line(points={{-120,40},{-71,40},{-71,40},
          {-21.2,40}}, color={0,0,127}));
  connect(T_oda, preTem.T) annotation (Line(points={{-120,40},{-34,40},{-34,18},
          {-21.2,18}}, color={0,0,127}));
  annotation (uses(AixLib(version="0.10.7"), Modelica(version="3.2.3")),
      experiment(StopTime=8640000, __Dymola_Algorithm="Dassl"));
end SimpleRoom;
